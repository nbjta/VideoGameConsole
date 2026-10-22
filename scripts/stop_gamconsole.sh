#!/usr/bin/env bash
# Nød-stop hvis OpenCV-vinduer hænger (lukker alle gamConsole-processer).
pkill -f "gamConsole" 2>/dev/null || true
pkill -f "gamConsole.builtin_game" 2>/dev/null || true
echo "gamConsole-processer stoppet (hvis nogen kørte)."
