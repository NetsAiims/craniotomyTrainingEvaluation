from PyQt5.QtWidgets import QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QMainWindow, QMessageBox
from PyQt5.QtCore import Qt
from src.utils.canvas import Canvas
import cv2

class Annotator(QMainWindow):

    def __init__(self, image = "src/images/iitd.png", parent = None, callback=lambda: None):
        super(Annotator, self).__init__(parent, Qt.WindowStaysOnTopHint)
        self.image = image
        self.callback = callback
        self.setup_ui()

    def save_image(self):
        try:
            if not self.imageBox.is_valid() or self.imageBox.get_dimensions() is None:
                print("Invalid bbox")
                self.show_error_dialog("Invalid Crop", "Please select a valid crop area and try again.")
                return False  
            
            dimensions = self.imageBox.get_dimensions()
            print(f"DEBUG: dimensions = {dimensions}")
            print(f"DEBUG: type of dimensions = {type(dimensions)}")
            
            if not isinstance(dimensions, tuple) or len(dimensions) == 0:
                print("Invalid bbox format")
                self.show_error_dialog("Invalid Crop", "Please select a valid crop area and try again.")
                return False
            
            print(f"DEBUG: type of first element = {type(dimensions[0])}")
            print(f"DEBUG: first element = {dimensions[0]}")
            
            # Check if each element is a tuple (coordinate pair) or a single value
            if isinstance(dimensions[0], tuple):
                # It's returning coordinate pairs like ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
                # Extract bounding box from coordinate pairs
                x_coords = [coord[0] for coord in dimensions]
                y_coords = [coord[1] for coord in dimensions]
                x = min(x_coords)
                y = min(y_coords)
                w = max(x_coords) - min(x_coords)
                h = max(y_coords) - min(y_coords)
            else:
                # It's returning (x, y, w, h) directly
                if len(dimensions) != 4:
                    print("Invalid bbox format - expected 4 values")
                    self.show_error_dialog("Invalid Crop", "Please select a valid crop area and try again.")
                    return False
                x, y, w, h = dimensions
            
            # Additional validation to ensure bbox is within image bounds
            image = cv2.imread(self.image)
            if image is None:
                print("Error: Could not load image")
                self.show_error_dialog("Error", "Could not load image. Please try again.")
                return False
                
            img_height, img_width = image.shape[:2]
            
            # Validate bbox coordinates
            if (x < 0 or y < 0 or w <= 0 or h <= 0 or 
                x + w > img_width or y + h > img_height):
                print(f"Invalid bbox: x={x}, y={y}, w={w}, h={h}, img_size=({img_width}, {img_height})")
                self.show_error_dialog("Invalid Crop", "Please select a valid crop area within the image bounds.")
                return False
            
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Prepare callback data
            if isinstance(dimensions[0], tuple):
                # Pass the original coordinate pairs
                callback_data = dimensions
            else:
                # Convert (x, y, w, h) to coordinate pairs if needed
                callback_data = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
            
            # Call the callback with error handling
            try:
                self.callback(img, callback_data)
                self.close()  # Only close if callback succeeds
                return True
            except Exception as e:
                print(f"Error in callback: {e}")
                self.show_error_dialog("Processing Error", "Failed to process the cropped image. Please try again.")
                return False
                
        except Exception as e:
            print(f"Error in save_image: {e}")
            self.show_error_dialog("Error", "An unexpected error occurred. Please try again.")
            return False

    def show_error_dialog(self, title, message):
        """Show error dialog and keep the annotator window open"""
        try:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle(title)
            msg.setText(message)
            msg.setStandardButtons(QMessageBox.Ok)
            
            # Set the message box to stay on top and be modal
            msg.setWindowFlags(msg.windowFlags() | Qt.WindowStaysOnTopHint)
            msg.setModal(True)
            
            result = msg.exec_()
            
            # After showing the dialog, ensure the annotator window stays active
            self.activateWindow()
            self.raise_()
            
            return result
        except Exception as e:
            print(f"Error showing dialog: {e}")

    def reset_canvas(self):
        # """Reset the canvas and ensure it's ready for new selections"""
        # try:
        #     # Reset the canvas
        #     self.imageBox.reset()
            
        #     # Ensure the canvas is properly refreshed and ready for interaction
        #     self.imageBox.update()
        #     self.imageBox.repaint()
            
        #     # Re-enable interaction if needed
        #     self.imageBox.setEnabled(True)
            
        #     # Ensure focus is on the canvas for mouse events
        #     self.imageBox.setFocus()
            
        # except Exception as e:
        #     print(f"Error resetting canvas: {e}")
        self.setup_ui()


    def setup_ui(self):
        self.mainWidget = QWidget()
        self.vbox = QVBoxLayout()
        self.imageBox = Canvas(self.image)
        self.vbox.addWidget(self.imageBox)
        
        self.buttons = [QPushButton("Reset (Ctrl+R)"), QPushButton("Crop Image")]
        
        # Connect reset button to our custom reset method
        self.buttons[0].clicked.connect(self.reset_canvas)
        self.buttons[1].clicked.connect(self.save_image)
        
        self.hlayout = QHBoxLayout()
        for item in self.buttons:
            self.hlayout.addWidget(item)
        self.vbox.addLayout(self.hlayout)
        self.mainWidget.setLayout(self.vbox)
        self.setCentralWidget(self.mainWidget)

    def closeEvent(self, event):
        """Handle window close event properly"""
        try:
            # Clean up canvas if needed
            if hasattr(self, 'imageBox'):
                self.imageBox.setParent(None)
            event.accept()
        except Exception as e:
            print(f"Error during close: {e}")
            event.accept()