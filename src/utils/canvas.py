
from PyQt5.QtGui import QImage, QPainter, QPen
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtCore import Qt
from src.utils.bbox import BoundingBox

class Canvas(QWidget):

    def __init__(self, photo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = QImage(photo)
        self.setFixedSize(self.image.width(), self.image.height())
        self.pressed = self.moving = False
        self.revisions = []
        self.bbox = None
        # Initialize caught flags
        self.caughtx = False
        self.caughty = False
        self.caughtw = False
        self.caughth = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.is_valid():
                if self.near(event.pos(), self.bbox.x):
                    self.caughtx = True
                if self.near(event.pos(), self.bbox.y):
                    self.caughty = True
                if self.near(event.pos(), self.bbox.w):
                    self.caughtw = True
                if self.near(event.pos(), self.bbox.h):
                    self.caughth = True

                self.pressed = True
                self.revisions.append(self.image.copy())
                self.center = event.pos()
                self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.moving = True
            flag = False
            if self.is_valid():
                if self.caughtx:
                    self.bbox.x = (event.pos().x(), event.pos().y()) 
                    flag = True
                elif self.caughty:
                    self.bbox.y = (event.pos().x(), event.pos().y())
                    flag = True
                elif self.caughtw:
                    self.bbox.w = (event.pos().x(), event.pos().y())
                    flag = True
                elif self.caughth:
                    self.bbox.h = (event.pos().x(), event.pos().y())
                    flag = True
                
                if flag:
                    l = []
                    l.append(self.bbox.x)
                    l.append(self.bbox.y)
                    l.append(self.bbox.w)
                    l.append(self.bbox.h)

                    self.reset()
                    self.revisions.append(self.image.copy())
                    qp = QPainter(self.image)
                    qp.setPen(QPen(Qt.blue, 3))
                    
                    self.bbox = BoundingBox(None, None, None, None)                
                    self.bbox.x = l[0]
                    self.bbox.y = l[1]
                    self.bbox.w = l[2]
                    self.bbox.h = l[3]

                    qp.drawLine(self.bbox.y[0], self.bbox.y[1], self.bbox.w[0], self.bbox.w[1])
                    qp.drawLine(self.bbox.x[0], self.bbox.x[1], self.bbox.y[0], self.bbox.y[1])
                    qp.drawLine(self.bbox.w[0], self.bbox.w[1], self.bbox.h[0], self.bbox.h[1])
                    qp.drawLine(self.bbox.x[0], self.bbox.x[1], self.bbox.h[0], self.bbox.h[1])
                    qp.drawEllipse(self.bbox.y[0]-5, self.bbox.y[1]-5, 10, 10)
                    qp.drawEllipse(self.bbox.h[0]-5, self.bbox.h[1]-5, 10, 10)
                    qp.drawEllipse(self.bbox.w[0]-5, self.bbox.w[1]-5, 10, 10)
                    qp.drawEllipse(self.bbox.x[0]-5, self.bbox.x[1]-5, 10, 10)

            self.update()

    def near(self, pos, corner):
        if corner is None:
            return False
        return abs(pos.x() - corner[0]) <= 10 and abs(pos.y() - corner[1]) <= 10

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.revisions.append(self.image.copy())
            
            if self.bbox is None:
                qp = QPainter(self.image)
                qp.setPen(QPen(Qt.blue, 3))
                self.bbox = BoundingBox((event.pos().x(), event.pos().y()), None, None, None)
                qp.drawEllipse(event.pos(), 3, 3)
                
            elif self.bbox.y is None:
                qp = QPainter(self.image)
                qp.setPen(QPen(Qt.blue, 3))
                self.bbox.y = (event.pos().x(), event.pos().y())
                qp.drawLine(self.bbox.x[0], self.bbox.x[1], self.bbox.y[0], self.bbox.y[1])
                qp.drawEllipse(event.pos(), 3, 3)
                
            elif self.bbox.w is None:
                qp = QPainter(self.image)
                qp.setPen(QPen(Qt.blue, 3))
                self.bbox.w = (event.pos().x(), event.pos().y())
                qp.drawLine(self.bbox.y[0], self.bbox.y[1], self.bbox.w[0], self.bbox.w[1])
                qp.drawLine(self.bbox.w[0], self.bbox.w[1], self.bbox.x[0], self.bbox.x[1])
                qp.drawEllipse(event.pos(), 3, 3)

            elif self.bbox.h is None:
                self.bbox.h = (event.pos().x(), event.pos().y())
                try:
                    l = []
                    l.append(self.bbox.x)
                    l.append(self.bbox.y)
                    l.append(self.bbox.w)
                    l.append(self.bbox.h)
                    
                    # Calculate center
                    cent1 = (l[0][0] + l[1][0] + l[2][0] + l[3][0]) / 4
                    cent2 = (l[0][1] + l[1][1] + l[2][1] + l[3][1]) / 4
                    
                    # Reset and rebuild bbox
                    self.reset()
                    self.bbox = BoundingBox(None, None, None, None)
                    
                    for i in l:
                        if self.bottomleft(i, (cent1, cent2)):
                            self.bbox.x = i
                        if self.bottomright(i, (cent1, cent2)):
                            self.bbox.y = i
                        if self.topright(i, (cent1, cent2)):
                            self.bbox.w = i
                        if self.topleft(i, (cent1, cent2)):
                            self.bbox.h = i
                    
                    # Validate that all corners are assigned
                    if None in [self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h]:
                        raise ValueError("Invalid bounding box configuration")
                    
                    self.revisions.append(self.image.copy())
                    qp = QPainter(self.image)
                    qp.setPen(QPen(Qt.blue, 3))

                    qp.drawLine(self.bbox.y[0], self.bbox.y[1], self.bbox.w[0], self.bbox.w[1])
                    qp.drawLine(self.bbox.x[0], self.bbox.x[1], self.bbox.y[0], self.bbox.y[1])
                    qp.drawLine(self.bbox.w[0], self.bbox.w[1], self.bbox.h[0], self.bbox.h[1])
                    qp.drawLine(self.bbox.x[0], self.bbox.x[1], self.bbox.h[0], self.bbox.h[1])
                    qp.drawEllipse(self.bbox.y[0]-5, self.bbox.y[1]-5, 10, 10)
                    qp.drawEllipse(self.bbox.h[0]-5, self.bbox.h[1]-5, 10, 10)
                    qp.drawEllipse(self.bbox.w[0]-5, self.bbox.w[1]-5, 10, 10)
                    qp.drawEllipse(self.bbox.x[0]-5, self.bbox.x[1]-5, 10, 10)
                    
                    self.caughtx = False
                    self.caughty = False
                    self.caughtw = False
                    self.caughth = False
                    
                except Exception as e:
                    print(f"Error creating bounding box: {e}")
                    
                    # Show message box
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Information)
                    msg.setText("Invalid Box")
                    msg.setInformativeText("Please select a valid box or close the window to take the entire image.\n Click reset to select from beginning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    
                    # Reset before showing message box to avoid state issues
                    self.bbox = None
                    self.caughtx = False
                    self.caughty = False
                    self.caughtw = False
                    self.caughth = False
                    
                    # Show message box
                    retval = msg.exec_()
                    
                    # Reset canvas after message box
                    self.reset()
                    print("Invalid bbox")
                 
            else:
                self.caughtx = False
                self.caughty = False
                self.caughtw = False
                self.caughth = False

            self.pressed = self.moving = False
            self.update()

    def bottomleft(self, point, center):
        return point[0] < center[0] and point[1] < center[1]
    
    def bottomright(self, point, center):
        return point[0] >= center[0] and point[1] < center[1]
        
    def topleft(self, point, center):
        return point[0] < center[0] and point[1] >= center[1]
        
    def topright(self, point, center):
        return point[0] >= center[0] and point[1] >= center[1]

    def paintEvent(self, event):
        qp = QPainter(self)
        rect = event.rect()
        qp.drawImage(rect, self.image, rect)

    def draw_point(self, qp):
        qp.setPen(QPen(Qt.blue, 15))
        qp.drawPoint(self.center)

    def draw_circle(self, qp):
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setPen(QPen(Qt.blue, 7, Qt.SolidLine))
        qp.drawEllipse(self.center, 3, 3)

    def draw_rectangle(self, qp, release_pos):
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setPen(QPen(Qt.blue, 5, Qt.SolidLine))
        w = release_pos.x() - self.center.x()
        h = release_pos.y() - self.center.y()
        self.bbox = BoundingBox(self.center.x(), self.center.y(), w, h)
        qp.drawRect(self.center.x(), self.center.y(), w, h)

    def undo(self):
        if self.revisions:
            self.image = self.revisions.pop()
            self.update()

    def reset(self):
        try:
            if self.revisions:
                self.image = self.revisions[0]
                self.revisions.clear()
                self.bbox = None
                self.caughtx = False
                self.caughty = False
                self.caughtw = False
                self.caughth = False
                self.update()
        except Exception as e:
            print(f"Error in reset(): {e}")
            # Fallback - just clear the bbox and flags
            self.bbox = None
            self.caughtx = False
            self.caughty = False
            self.caughtw = False
            self.caughth = False
    
    def is_valid(self):
        if self.bbox is None or None in [self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h]:
            return False
        return True
    
    def get_dimensions(self):
        if self.bbox is None:
            return None
        return self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h