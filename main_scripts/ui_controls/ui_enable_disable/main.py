import configparser
import os
import subprocess
import sys

def main():
    """
    Main function to determine which script to run based on the configuration file.
    """
    config_file = 'config.ini'
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_file)

    try:
        ui_enabled = config.getboolean('SETTINGS', 'ui_enable')
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"Error reading configuration: {e}")
        print("Please ensure 'ui_enable' is set under the '[SETTINGS]' section.")
        sys.exit(1)

    if ui_enabled:
        print("UI is enabled. Starting the UI application...")
        try:
            # We are calling ui.py as a separate process.
            subprocess.run([sys.executable, 'ui.py'])
        except FileNotFoundError:
            print("Error: ui.py not found. Please make sure the file exists in the same directory.")
    else:
        print("UI is disabled. Starting the Driver Monitoring System...")
        try:
            # We are calling withoutui.py as a separate process.
            subprocess.run([sys.executable, 'withoutui.py'])
        except FileNotFoundError:
            print("Error: withoutui.py not found. Please make sure the file exists in the same directory.")

if __name__ == '__main__':
    main()

