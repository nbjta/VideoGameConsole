#5
"""WebKit-vindue i gamConsole. Siden får URL'en direkte via loadRequest."""

from __future__ import annotations

import json
import time
from typing import Optional

from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyRegular,
    NSBackingStoreBuffered,
    NSColor,
    NSEventMaskAny,
    NSViewHeightSizable,
    NSViewWidthSizable,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
)
from CoreFoundation import CFRunLoopRunInMode, kCFRunLoopDefaultMode
from Foundation import NSDate, NSDefaultRunLoopMode, NSURL, NSURLRequest
from WebKit import (
    WKPermissionDecisionGrant,
    WKUserScript,
    WKUserScriptInjectionTimeAtDocumentStart,
    WKWebView,
    WKWebViewConfiguration,
)

# Lobbyens --bg-base, så vinduet ikke er hvidt mens siden hentes.
_LOBBY_BG = (4 / 255.0, 6 / 255.0, 10 / 255.0, 1.0)
_PAGE_JS = (
    "(function(){"
    "var root=document.getElementById('root');"
    "var errors=window.__gamconsoleErrors||[];"
    "return JSON.stringify({root:root?root.childNodes.length:0, errors:errors.slice(0,4)});"
    "})()"
)


def _make_delegate_class():
    import objc
    from AppKit import NSObject

    class _Delegate(NSObject):
        def init(self):
            self = objc.super(_Delegate, self).init()
            if self is None:
                return None
            self.browser = None
            return self

        def webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_(
            self, _web, _origin, _frame, _media_type, decision
        ):
            decision(WKPermissionDecisionGrant)

        def webView_didFinishNavigation_(self, web, _navigation):
            if self.browser is not None:
                url = web.URL()
                self.browser.current_url = "" if url is None else str(url.absoluteString())
                self.browser.loaded_at = time.monotonic()

        def webView_didFailNavigation_withError_(self, _web, _navigation, error):
            if self.browser is not None:
                self.browser.error = str(error.localizedDescription())

        def webView_didFailProvisionalNavigation_withError_(self, _web, _navigation, error):
            if self.browser is not None:
                self.browser.error = str(error.localizedDescription())

        def windowWillClose_(self, _notification):
            if self.browser is not None:
                self.browser.closed = True

    return _Delegate


_DelegateClass = _make_delegate_class()


class ConsoleBrowser:
    """Et vindue med én side. GamConsole ejer vinduet, ikke Safari eller Chrome."""

    def __init__(self, url: str, title: str = "gamConsole") -> None:
        self.url = url
        self.title = title
        self.current_url = ""
        self.error: Optional[str] = None
        self.closed = False
        self.loaded_at: Optional[float] = None
        self._window = None
        self._web = None
        self._delegate = None
        self._poll_pending = False
        self._last_poll = 0.0

    def show(self) -> bool:
        if self._window is not None and not self.closed:
            self._window.makeKeyAndOrderFront_(None)
            return True
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable
        )
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            ((80, 80), (1100, 740)),
            style,
            NSBackingStoreBuffered,
            False,
        )
        window.setTitle_(self.title)
        window.setReleasedWhenClosed_(False)
        config = WKWebViewConfiguration.alloc().init()
        #5 Fang JS-fejl før React starter, så en tom side kan vises i statusfeltet.
        script = WKUserScript.alloc().initWithSource_injectionTime_forMainFrameOnly_(
            "window.__gamconsoleErrors=[];"
            "window.addEventListener('error',function(e){"
            "window.__gamconsoleErrors.push(String(e.message));"
            "});",
            WKUserScriptInjectionTimeAtDocumentStart,
            True,
        )
        config.userContentController().addUserScript_(script)
        web = WKWebView.alloc().initWithFrame_configuration_(((0, 0), (1100, 740)), config)
        web.setUnderPageBackgroundColor_(NSColor.colorWithSRGBRed_green_blue_alpha_(*_LOBBY_BG))
        web.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        delegate = _DelegateClass.alloc().init()
        delegate.browser = self
        web.setUIDelegate_(delegate)
        web.setNavigationDelegate_(delegate)
        window.setDelegate_(delegate)
        window.setContentView_(web)
        nsurl = NSURL.URLWithString_(self.url)
        if nsurl is None:
            self.error = f"Ugyldig adresse: {self.url}"
            return False
        web.loadRequest_(NSURLRequest.requestWithURL_(nsurl))
        window.makeKeyAndOrderFront_(None)
        app.activateIgnoringOtherApps_(True)
        self._window = window
        self._web = web
        self._delegate = delegate
        self.closed = False
        self.loaded_at = None
        return True

    def pump(self) -> None:
        if self._window is None or self.closed:
            return
        #5 nextEvent alene kører ikke WebKits netværk og tegning. Så bliver siden hvid.
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.02, False)
        app = NSApplication.sharedApplication()
        for _ in range(40):
            event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                NSEventMaskAny,
                NSDate.distantPast(),
                NSDefaultRunLoopMode,
                True,
            )
            if event is None:
                break
            app.sendEvent_(event)
        app.updateWindows()
        self._poll_page()

    def _poll_page(self) -> None:
        now = time.monotonic()
        if self._web is None or self._poll_pending or now - self._last_poll < 0.4:
            return
        self._last_poll = now
        self._poll_pending = True

        def done(value, _error) -> None:
            self._poll_pending = False
            if self.loaded_at is None or self.error or not isinstance(value, str):
                return
            if time.monotonic() - self.loaded_at < 1.5:
                return
            try:
                payload = json.loads(value)
            except json.JSONDecodeError:
                return
            errors = payload.get("errors") or []
            if payload.get("root") == 0 and errors:
                self.error = "Lobbyen tegnede ikke. " + str(errors[0])

        self._web.evaluateJavaScript_completionHandler_(_PAGE_JS, done)

    def close(self) -> None:
        window = self._window
        self._window = None
        self._web = None
        self._delegate = None
        self.closed = True
        if window is not None:
            window.setDelegate_(None)
            window.orderOut_(None)
            window.close()
