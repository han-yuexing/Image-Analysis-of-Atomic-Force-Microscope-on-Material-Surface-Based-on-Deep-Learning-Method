#include "mainwindow.h"
#include "ui_mainwindow.h"
#if QT_VERSION >= 0x05000
#include <QtWidgets/QMainWindow>
#else
#include <QtGui/QMainWindow>
#endif
#include <QFileDialog>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <pthread.h>
#include "python2.7/Python.h"
#define MAXPATH 1000

using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_OpenImage_clicked()
{
    QString img_name = QFileDialog::getOpenFileName( this, tr("Open Image"), ".", tr("Image Files(*.png *.jpg *.jpeg *.bmp)"));
    filename = img_name;
    src = cv::imread(img_name.toUtf8().data());
    if (!src.data) {
          return;
    }

    QImage image;
    Mat rgb;
    if (src.channels() == 3) {
          cvtColor( src,  rgb, CV_BGR2RGB);
          image = QImage( (const unsigned char*)(rgb.data), rgb.cols, rgb.rows, rgb.cols * rgb.channels(), QImage::Format_RGB888 );
    } else {
         image = QImage( (const unsigned char*)(src.data), src.cols, src.rows, src.cols * src.channels(), QImage::Format_Indexed8);
    }

    ui->showImage->setPixmap( QPixmap::fromImage(image) );
    ui->showImage->resize( ui->showImage->pixmap()->size() );
}

void MainWindow::on_closeButton_clicked()
{
    close();
}

void MainWindow::on_LoG_clicked()
{
    if (!src.data) {
        QMessageBox message(QMessageBox::NoIcon, "Warning", "No Source DNA Picture.");
        message.exec();
        qDebug("No Picture");
        return;
    }

    string s_filename = filename.toUtf8().data();
    //string s_filename = "/home/hadoop/project/QT/denoise/5.png";
    char buffer[MAXPATH];
    getcwd(buffer, MAXPATH);

    string s_dir = s_filename.substr(0, s_filename.rfind('/', -1));
    //cout<<s_dir<<endl;

    string s_file = s_filename.substr(s_filename.rfind('/', -1) + 1);
    //cout<<s_file<<endl;

    string file = s_file.substr(0, s_file.find('.', 0));

    outfile = string("image_") + file + string(".png");
    //cout<<file<<endl;

    string copyImg_exec = string("cp -rf ") + s_filename + string(" ") + string(buffer);
    //cout<<copyImg_exec<<endl;
    popen(copyImg_exec.c_str(), "r");

    QString max_sigma_Q=ui->lineEdit->text();
    string max_sigma_p=max_sigma_Q.toStdString();

    string object_exec_log = string("python ") + s_dir + string("/") + string("plot_blob.py ") + file + string(" ") + max_sigma_p;  //---add parameters-----------

    //cout<<object_exec_log<<endl;
    popen(object_exec_log.c_str(), "r");
    //popen("python plot_blob.py 5", "r");

    sleep(0.5);
    QMessageBox message(QMessageBox::NoIcon, "Info", "LoG Operator is OK.");
    message.exec();
    qDebug("Info");
}

void MainWindow::on_segmentation_clicked()
{
    if (!src.data) {
        QMessageBox message(QMessageBox::NoIcon, "Warning", "No Source DNA Picture.");
        message.exec();
        qDebug("No Picture");
        return;
    }

	IMG_PROC *solution=new IMG_PROC();
	solution->show_src(src);                   //读取图片
	solution->avgRGB();                        //计算平均像素值
    if (outfile.empty()) {
        QMessageBox message(QMessageBox::NoIcon, "Warning", "Muse first exec LoG button.");
        message.exec();
        qDebug("No LoG Filted Image");
        return;
    }
    solution->remove_areanoise_Tong(outfile);
	solution->remove_watermark(); //去除水印
	solution->get_binarysrc();    //二值化
	solution->new_erode();      //腐蚀
	solution->new_dilate();     //膨胀：填充连通域
	Mat dst = solution->filling_pic(); //图像背景填充


	 Mat out;
	 QImage image;
	 out = dst.clone();
	 if (dst.channels() == 3) {
         cvtColor(out,  out, CV_BGR2RGB);
		 image = QImage( (const unsigned char*)(out.data), out.cols, out.rows, out.cols * out.channels(), QImage::Format_RGB888 );
	 } else {
		 image = QImage( (const unsigned char*)(dst.data), dst.cols, dst.rows, dst.cols * dst.channels(), QImage::Format_Indexed8);
	 }

     image = image.scaled(ui->showImage->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
     ui->showImage->setPixmap( QPixmap::fromImage(image) );
     ui->showImage->resize( ui->showImage->pixmap()->size() );

}
