import sys
import joblib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget,
    QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QLineEdit, QFormLayout, QMessageBox, QSizePolicy
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


def load_scores():
    """Regresyon ve sınıflandırma skorlarını txt dosyalarından oku"""
    scores = {"Regresyon": {}, "Sınıflandırma": {}}

    try:
        with open("skor/regression_models_score.txt", "r", encoding="utf-8") as f:
            block = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not line.startswith(("MSE", "RMSE", "MAE", "R²")):
                    block = line
                    scores["Regresyon"][block] = {}
                else:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key, val = parts[0].strip(), float(parts[1])
                        scores["Regresyon"][block][key] = val
    except FileNotFoundError:
        print("⚠ regression_models_score.txt bulunamadı")

    try:
        with open("skor/classification_models_score.txt", "r", encoding="utf-8") as f:
            block = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "Confusion" not in line and not line.startswith(("Accuracy", "F1")):
                    block = line
                    scores["Sınıflandırma"][block] = {}
                elif line.startswith("Accuracy"):
                    scores["Sınıflandırma"][block]["Accuracy"] = float(line.split(":")[1])
                elif line.startswith("F1"):
                    scores["Sınıflandırma"][block]["F1"] = float(line.split(":")[1])
    except FileNotFoundError:
        print("⚠ classification_models_score.txt bulunamadı")

    return scores


class MLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OptiModel - ML Karşılaştırma ve Tahmin Arayüzü")
        self.setGeometry(200, 200, 1000, 600)

        layout = QVBoxLayout()
        tabs = QTabWidget()

        self.tab_compare = QWidget()
        self.tab_predict = QWidget()

        tabs.addTab(self.tab_compare, "Model Karşılaştırma")
        tabs.addTab(self.tab_predict, "Tahmin Yap")

        layout.addWidget(tabs)
        self.setLayout(layout)

        # MODELLERİ YÜKLE
        self.models = {
            "LinearRegression": joblib.load(r"C:\Users\Batu\Documents\optimodel\models\LinearRegression.joblib"),
            "Ridge": joblib.load(r"C:\Users\Batu\Documents\optimodel\models\Ridge.joblib"),
            "DecisionTreeRegressor": joblib.load(r"C:\Users\Batu\Documents\optimodel\models\DecisionTreeRegressor.joblib"),
            "RandomForestRegressor": joblib.load(r"C:\Users\Batu\Documents\optimodel\models\RandomForestRegressor.joblib"),
            "SVR": joblib.load(r"C:\Users\Batu\Documents\optimodel\models\SVR.joblib"),
            "PolynomialRegression": joblib.load(r"C:\Users\Batu\Documents\optimodel\models\PolynomialRegression.joblib"),
            "LogisticRegression": joblib.load(r"C:\Users\Batu\Documents\optimodel\models\LogisticRegression.joblib"),
            "DecisionTreeClassifier": joblib.load(r"C:\Users\Batu\Documents\optimodel\models\DecisionTreeClassifier.joblib"),
            "RandomForestClassifier": joblib.load(r"C:\Users\Batu\Documents\optimodel\models\RandomForestClassifier.joblib"),
            "KNeighborsClassifier": joblib.load(r"C:\Users\Batu\Documents\optimodel\models\KNeighborsClassifier.joblib"),
            "ExtraTreesClassifier": joblib.load(r"C:\Users\Batu\Documents\optimodel\models\ExtraTreesClassifier.joblib")
        }

        self.scores = load_scores()

        self.init_compare_tab()
        self.init_predict_tab()

    # ------------------- TAB 1: KARŞILAŞTIRMA -------------------
    def init_compare_tab(self):
        layout = QVBoxLayout()

        self.task_combo = QComboBox()
        self.task_combo.addItems(["Regresyon", "Sınıflandırma"])
        self.task_combo.currentTextChanged.connect(self.update_metrics)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["MSE", "RMSE", "MAE", "R²"])
        
        btn_show = QPushButton("Karşılaştırmayı Göster")
        btn_show.clicked.connect(self.show_comparison)

        self.table = QTableWidget()
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.table.setMaximumHeight(200)  # tablo çok yer kaplamasın

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(QLabel("Görev Tipi Seç:"))
        layout.addWidget(self.task_combo)
        layout.addWidget(QLabel("Metrik Seç:"))
        layout.addWidget(self.metric_combo)
        layout.addWidget(btn_show)
        layout.addWidget(self.table)
        layout.addWidget(self.canvas)

        self.tab_compare.setLayout(layout)

    def update_metrics(self, task):
        if task == "Regresyon":
            self.metric_combo.clear()
            self.metric_combo.addItems(["MSE", "RMSE", "MAE", "R²"])
        else:
            self.metric_combo.clear()
            self.metric_combo.addItems(["Accuracy", "F1"])
            
    def show_comparison(self):
        task = self.task_combo.currentText()
        metric = self.metric_combo.currentText()
        data = self.scores.get(task, {})

        self.table.setRowCount(len(data))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Model", metric])

        values, labels = [], []
        for row, (model, metrics) in enumerate(data.items()):
            value = metrics.get(metric, None)
            if value is not None:
                self.table.setItem(row, 0, QTableWidgetItem(model))
                self.table.setItem(row, 1, QTableWidgetItem(str(round(value, 4))))
                labels.append(model)
                values.append(value)

        # --- Grafik Düzenleme ---
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar(labels, values)
        ax.set_title(f"{task} - {metric} Karşılaştırması")
        ax.set_ylabel(metric)

        # X label düzeltme
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")

        # Alt boşluğu genişlet
        self.figure.subplots_adjust(bottom=0.2)

        self.canvas.draw()

    # ------------------- TAB 2: PREDICT -------------------
    def init_predict_tab(self):
        layout = QVBoxLayout()

        self.task_combo_pred = QComboBox()
        self.task_combo_pred.addItems(["Regresyon", "Sınıflandırma"])
        self.task_combo_pred.currentTextChanged.connect(self.update_model_list)

        self.model_combo = QComboBox()
        self.form_layout = QFormLayout()
        self.input_fields = {}

        self.btn_predict = QPushButton("Tahmin Yap")
        self.btn_predict.clicked.connect(self.make_prediction)

        layout.addWidget(QLabel("Görev Tipi Seç:"))
        layout.addWidget(self.task_combo_pred)
        layout.addWidget(QLabel("Model Seç:"))
        layout.addWidget(self.model_combo)
        layout.addLayout(self.form_layout)
        layout.addWidget(self.btn_predict)

        self.tab_predict.setLayout(layout)
        self.update_model_list("Regresyon")

    def update_model_list(self, task):
        self.model_combo.clear()
        if task == "Regresyon":
            self.model_combo.addItems([
                "LinearRegression", "Ridge", "DecisionTreeRegressor",
                "RandomForestRegressor", "SVR", "PolynomialRegression"
            ])
            columns = ["ort_gelir", "ev_yas", "oda_sayisi",
                       "yatak_odasi"]
        else:
            self.model_combo.addItems([
                "LogisticRegression", "DecisionTreeClassifier",
                "RandomForestClassifier", "KNeighborsClassifier", "ExtraTreesClassifier"
            ])
            columns = ["alkol", "malik_asid_miktari", "kül_miktarı",
                       "kül_alkalinitesi", "magnezyum_miktari", "toplam_fenol",
                       "flavonoid_fenol", "flavonoid_olmayan_fenoller",
                       "proantosiyanin_miktari", "renk_yogunlugu",
                       "renk_tonu", "prolin_miktari"]

        # reset inputs
        for i in reversed(range(self.form_layout.count())):
            self.form_layout.itemAt(i).widget().deleteLater()
        self.input_fields = {}

        for col in columns:
            line_edit = QLineEdit()
            self.form_layout.addRow(QLabel(col), line_edit)
            self.input_fields[col] = line_edit

    def make_prediction(self):
        model_name = self.model_combo.currentText()
        model = self.models.get(model_name, None)

        if isinstance(model, dict):
            model = model.get("model", None)

        if model is None or not hasattr(model, "predict"):
            QMessageBox.warning(self, "Hata", f"{model_name} modeli yüklenemedi!")
            return

        try:
            input_data = [float(self.input_fields[c].text()) for c in self.input_fields]
        except ValueError:
            QMessageBox.warning(self, "Hata", "Tüm alanlara sayısal değer girin!")
            return

        X_new = np.array(input_data).reshape(1, -1)

        try:
            pred = model.predict(X_new)
            QMessageBox.information(self, "Tahmin Sonucu", f"{model_name} Tahmin: {pred[0]}")
        except Exception as e:
            QMessageBox.critical(self, "Tahmin Hatası", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLApp()
    window.show()
    sys.exit(app.exec_())
