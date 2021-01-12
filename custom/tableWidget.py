from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class TableWidget(QTableWidget):
    def __init__(self, parent=None):
        super(TableWidget, self).__init__(parent=parent)
        self.mainwindow = parent
        self.setShowGrid(True)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().sectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().sectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setStretchLastSection(True)
        self.setFocusPolicy(Qt.NoFocus)

    def signal_connect(self):
        for spinbox in self.findChildren(QSpinBox):
            spinbox.valueChanged.connect(self.update_item)
        for doublespinbox in self.findChildren(QDoubleSpinBox):
            doublespinbox.valueChanged.connect(self.update_item)
        for combox in self.findChildren(QComboBox):
            combox.currentIndexChanged.connect(self.update_item)
        for checkbox in self.findChildren(QCheckBox):
            checkbox.stateChanged.connect(self.update_item)

    def update_item(self):
        param = self.get_params()
        self.mainwindow.useListWidget.currentItem().update_params(param)
        self.mainwindow.update_image()

    def update_params(self, param=None):
        for key in param.keys():
            box = self.findChild(QWidget, name=key)
            if isinstance(box, QSpinBox) or isinstance(box, QDoubleSpinBox):
                box.setValue(param[key])
            elif isinstance(box, QComboBox):
                box.setCurrentIndex(param[key])
            elif isinstance(box, QCheckBox):
                box.setChecked(param[key])

    def get_params(self):
        param = {}
        for spinbox in self.findChildren(QSpinBox):
            param[spinbox.objectName()] = spinbox.value()
        for doublespinbox in self.findChildren(QDoubleSpinBox):
            param[doublespinbox.objectName()] = doublespinbox.value()
        for combox in self.findChildren(QComboBox):
            param[combox.objectName()] = combox.currentIndex()
        for combox in self.findChildren(QCheckBox):
            param[combox.objectName()] = combox.isChecked()
        return param


class ExpTranWidget(TableWidget):
    def __init__(self, parent=None):
        super(ExpTranWidget, self).__init__(parent=parent)

        self.param1_spinBox = QDoubleSpinBox()
        self.param1_spinBox.setMinimum(0)
        self.param1_spinBox.setMaximum(3)
        self.param1_spinBox.setSingleStep(0.1)
        self.param1_spinBox.setObjectName('param1')

        self.param2_spinbox = QSpinBox()
        self.param2_spinbox.setMinimum(0)
        self.param2_spinbox.setSingleStep(1)
        self.param2_spinbox.setObjectName('param2')

        self.setColumnCount(2)
        self.setRowCount(2)

        self.setItem(0, 0, QTableWidgetItem('补偿系数'))
        self.setCellWidget(0, 1, self.param1_spinBox)
        self.setItem(1, 0, QTableWidgetItem('Gamma'))
        self.setCellWidget(1, 1, self.param2_spinbox)
        self.signal_connect()


class GammaITabelWidget(TableWidget):
    def __init__(self, parent=None):
        super(GammaITabelWidget, self).__init__(parent=parent)
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0)
        self.gamma_spinbox.setSingleStep(0.1)
        self.gamma_spinbox.setObjectName('gamma')

        self.setColumnCount(2)
        self.setRowCount(1)

        self.setItem(0, 0, QTableWidgetItem('gamma'))
        self.setCellWidget(0, 1, self.gamma_spinbox)
        self.signal_connect()


class FilterTabledWidget(TableWidget):
    def __init__(self, parent=None):
        super(FilterTabledWidget, self).__init__(parent=parent)

        self.kind_comBox = QComboBox()
        self.kind_comBox.addItems(['均值滤波', '高斯滤波', '中值滤波'])
        self.kind_comBox.setObjectName('kind')

        self.ksize_spinBox = QSpinBox()
        self.ksize_spinBox.setObjectName('ksize')
        self.ksize_spinBox.setMinimum(1)
        self.ksize_spinBox.setSingleStep(2)

        self.setColumnCount(2)
        self.setRowCount(2)
        self.setItem(0, 0, QTableWidgetItem('类型'))
        self.setCellWidget(0, 1, self.kind_comBox)
        self.setItem(1, 0, QTableWidgetItem('核大小'))
        self.setCellWidget(1, 1, self.ksize_spinBox)

        self.signal_connect()


class HisBalanceWidget(TableWidget):
    def __init__(self, parent=None):
        super(HisBalanceWidget, self).__init__(parent=parent)


class ImcompWidget(TableWidget):
    def __init__(self, parent=None):
        super(ImcompWidget, self).__init__(parent=parent)


class LaplaceSharpWidget(TableWidget):
    def __init__(self, parent=None):
        super(LaplaceSharpWidget, self).__init__(parent=parent)


class BorderDetectWidget(TableWidget):
    def __init__(self, parent=None):
        super(BorderDetectWidget, self).__init__(parent=parent)
        self.kind_comBox = QComboBox()
        self.kind_comBox.addItems(['Scharr算子', '拉普拉斯算子', 'Canny算子'])
        self.kind_comBox.setObjectName('kind')

        self.setColumnCount(2)
        self.setRowCount(1)
        self.setItem(0, 0, QTableWidgetItem('类型'))
        self.setCellWidget(0, 1, self.kind_comBox)

        self.signal_connect()


class DFTWidget(TableWidget):
    def __init__(self, parent=None):
        super(DFTWidget, self).__init__(parent=parent)


class HighPassFilterWidget(TableWidget):
    def __init__(self, parent=None):
        super(HighPassFilterWidget, self).__init__(parent=parent)
        self.d_spinbox = QDoubleSpinBox()
        self.d_spinbox.setMinimum(10)
        self.d_spinbox.setSingleStep(5)
        self.d_spinbox.setObjectName('d')

        self.setColumnCount(2)
        self.setRowCount(1)

        self.setItem(0, 0, QTableWidgetItem('d'))
        self.setCellWidget(0, 1, self.d_spinbox)
        self.signal_connect()


class LowPassFilterWidget(TableWidget):
    def __init__(self, parent=None):
        super(LowPassFilterWidget, self).__init__(parent=parent)
        self.d_spinbox = QDoubleSpinBox()
        self.d_spinbox.setMinimum(10)
        self.d_spinbox.setSingleStep(5)
        self.d_spinbox.setObjectName('d')

        self.setColumnCount(2)
        self.setRowCount(1)

        self.setItem(0, 0, QTableWidgetItem('d'))
        self.setCellWidget(0, 1, self.d_spinbox)
        self.signal_connect()


class MakeBlurredWidget(TableWidget):
    def __init__(self, parent=None):
        super(MakeBlurredWidget, self).__init__(parent=parent)


class ReverseFilterWidget(TableWidget):
    def __init__(self, parent=None):
        super(ReverseFilterWidget, self).__init__(parent=parent)


class WienerFilterWidget(TableWidget):
    def __init__(self, parent=None):
        super(WienerFilterWidget, self).__init__(parent=parent)
