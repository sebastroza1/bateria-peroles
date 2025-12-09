import sys
from PyQt5.QtWidgets import QApplication
from ui_logic.interface_controller import InterfaceController

def main():
    app = QApplication(sys.argv)
    window = InterfaceController()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()