from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from config import items


class MyListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.mainwindow = parent
        self.setDragEnabled(True)
        self.setFocusPolicy(Qt.NoFocus)


class UsedListWidget(MyListWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setAcceptDrops(True)
        self.setFlow(QListView.TopToBottom)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.itemClicked.connect(self.show_attr)
        self.setMinimumWidth(200)
        self.move_item = None

    def contextMenuEvent(self, e):
        item = self.itemAt(self.mapFromGlobal(QCursor.pos()))
        if not item: return
        menu = QMenu()
        delete_action = QAction('删除', self)
        delete_action.triggered.connect(lambda: self.delete_item(item))
        menu.addAction(delete_action)
        menu.exec(QCursor.pos())

    def delete_item(self, item):
        self.takeItem(self.row(item))
        self.mainwindow.update_image()
        self.mainwindow.dock_attr.close()

    def dropEvent(self, event):
        super().dropEvent(event)
        self.mainwindow.update_image()

    def show_attr(self):
        item = self.itemAt(self.mapFromGlobal(QCursor.pos()))
        if not item: return
        param = item.get_params()
        if type(item) in items:
            index = items.index(type(item))
            self.mainwindow.stackedWidget.setCurrentIndex(index)
            self.mainwindow.stackedWidget.currentWidget().update_params(param)
            self.mainwindow.dock_attr.show()


class FuncListWidget(MyListWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedHeight(124)
        self.setFlow(QListView.LeftToRight)
        self.setViewMode(QListView.IconMode)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAcceptDrops(False)
        for itemType in items:
            self.addItem(itemType())
        self.itemClicked.connect(self.add_used_function)

    def add_used_function(self):
        func_item = self.currentItem()
        if type(func_item) in items:
            use_item = type(func_item)()
            self.mainwindow.useListWidget.addItem(use_item)
            self.mainwindow.update_image()

    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        self.setCurrentRow(-1)
