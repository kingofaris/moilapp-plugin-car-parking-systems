# moilapp-plugin-car-parking-systems
*Last Update: Agust 9, 2024*

### *Create by*

* Abdul Aziz
* I Made Adhika Dananjaya 
* Rega Arzula Akbar 

### Introduction

Moilapp plugin car parking system is a plugin designed as an example for creating new Moilapp plugins. This plugin was created with the aim of making it easier for an institution to monitor vehicles entering or leaving a place in the parking area. This plugin will record the vehicle license plate, the time the vehicle enters or leaves the parking area, and the number of cars entering or leaving the parking area.


### Requirements
```
easyocr==1.7.1
PyQt6==6.3.1
pytesseract==0.3.10
pyqt6-tools==6.3.1.3.3
torch==2.3.1
torchvision==0.18.1
ultralytics==8.2.60
```

### How to Run | Usage
To use this plugin, follow these steps :
1. Open the Moilapp directory in the terminal.
2. Build the virtual environment:
```
$ virtualenv venv
```
3. Activated the virtual environment:
```
$ source venv/bin/activate
```
4. Navigate to the plugin directory:
```
$ cd src/plugins
```
5. Clone the repository :
```
$ git clone https://github.com/Herusyahputra/moilapp-plugin-car-parking-systems.git
```
5. Navigate to car parking plugin:
```
$ cd moilapp-plugin-car-parking-systems
```
8. Install requirements:
```
$ pip install -r requirements.txt
```
9. Back to the *src* directory:
```
$ cd ../../
```
10. Run the Moilapp:
```
$ python3 main.py
```
11. select the plugin that has been marked:
![](img_plugin/plugin_view.png)
![](img_plugin/plugin_img.jpeg)

### Contact
For any questions, suggestions, or concerns regarding the Plugin Application, please feel free to contact the repository owner at herusyahputra@telkomuniversity.ac.id. 
