
"""
Jetson Nano LED blink test
Usage:
    python3 led_test.py [period_seconds]

Default period is 1 s ON, 1 s OFF (i.e., period = 1.0).
"""

import Jetson.GPIO as GPIO
import time
import sys
import signal

# ---------- Configuration ----------
LED_PIN_BOARD = 32      # Physical pin number (GPIO12)
DEFAULT_PERIOD = 1.0    # Seconds
# -----------------------------------

def cleanup(signum=None, frame=None):
    """Return GPIO lines to a safe state and exit."""
    GPIO.output(LED_PIN_BOARD, GPIO.LOW)
    GPIO.cleanup()
    print("\nGPIO cleaned up. Bye!")
    sys.exit(0)

def main():
    # Parse optional period argument
    try:
        period = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PERIOD
        if period <= 0:
            raise ValueError
    except (ValueError, IndexError):
        print("Invalid period; using default:", DEFAULT_PERIOD)
        period = DEFAULT_PERIOD

    # Register Ctrl‑C handler
    signal.signal(signal.SIGINT, cleanup)

    # Set up the GPIO pin
    GPIO.setmode(GPIO.BOARD)          # Use physical pin numbering
    GPIO.setup(LED_PIN_BOARD, GPIO.OUT, initial=GPIO.LOW)

    print(f"Blinking LED on pin {LED_PIN_BOARD} every {period}s "
          "(Ctrl‑C to stop)")
    try:
        while True:
            GPIO.output(LED_PIN_BOARD, GPIO.HIGH)
            time.sleep(period)
            GPIO.output(LED_PIN_BOARD, GPIO.LOW)
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

if __name__ == "__main__":
    main()


'''


import Jetson.GPIO as GPIO
import time

# Use BCM numbering
GPIO.setmode(GPIO.BOARD)  # or GPIO.BCM if using GPIO numbers directly

led_pin = 12  # Physical Pin 12 = GPIO18

# Set up pin as output
GPIO.setup(led_pin, GPIO.OUT)

# Blink LED
print("Blinking LED... Press Ctrl+C to stop.")
try:
    while True:
        GPIO.output(led_pin, GPIO.HIGH)  # LED ON
        time.sleep(1)
        GPIO.output(led_pin, GPIO.LOW)   # LED OFF
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
finally:
    GPIO.cleanup()  # Reset GPIO state

'''