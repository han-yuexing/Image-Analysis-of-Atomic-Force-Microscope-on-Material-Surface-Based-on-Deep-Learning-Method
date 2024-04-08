#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include<QMessageBox>
#include <QTimer>
#include <cctype>
#include <QProgressDialog>
#include "IMG_PROC.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
private:
    void LoG();
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
private slots:
    void on_OpenImage_clicked();

    void on_closeButton_clicked();

    void on_segmentation_clicked();

    void on_LoG_clicked();

private:
    Mat src;
    QString filename;
    String outfile;
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
