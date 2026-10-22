import os


def main() -> None:
    ui = os.environ.get("GAMCONSOLE_UI", "opencv").strip().lower()
    if ui == "tk":
        from gamConsole.menu import run_menu

        run_menu()
    else:
        from gamConsole.menu_opencv import run_opencv_menu

        run_opencv_menu()


if __name__ == "__main__":
    main()
