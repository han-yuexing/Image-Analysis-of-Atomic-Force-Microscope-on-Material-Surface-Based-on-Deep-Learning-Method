QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = DNA_denoise
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    IMG_PROC.cpp

HEADERS  += mainwindow.h \
    IMG_PROC.h

FORMS    += mainwindow.ui

CONFIG += C++11

INCLUDEPATH += /usr/local/include \
                              /usr/local/include/opencv \
                              /usr/local/include/opencv2 \


LIBS += /usr/local/lib/libopencv_*.so \

