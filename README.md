# Masterproef
Masterproef - Mobile Deep Visual Detection and Recognition

Deze repository bevat al de code waarnaar verwezen is in de masterproef. In de folder code is alles opgedeeld met dezelfde structuur als de hoofdstukken in de tekst. Elk deel bevat een Android applicatie en een bijhorende Google Colab Notebook. Ook is in de naam van de folder dezelfde hoofdstuknummering terug te vinden als de tekst.

Om de Android applicaties uit te voeren moet eerst het model gegenereerd worden met de bijhorende Google Colab notebook.

## Opdeling code
Code/Herkenningssystemen/ResNet50/PyTorch:
- Bevat de PyTorch code van de ResNet50 implementatie.
- Het genereerde model moet aan de assets folder van het Android project worden toegevoegd.

Code/Herkenningssystemen/ResNet50/TensorFlow:
- Bevat de TensorFlow code van de ResNet50 implementatie.
- Het genereerde model moet als TensorFlow Lite model aan het Android project worden toegevoegd.

Code/Herkenningssystemen/ResNet50/ONNX:
- Bevat de ONNX code van de ResNet50 implementatie.
- De ONNX modellen worden gegenereerd in de TensorFlow en PyTorch notebook van ResNet50.
- Het genereerde model moet aan de res/raw folder van het Android project worden toegevoegd.

Code/Detectiesystemen/Faster_RCNN/PyTorch:
- Bevat de PyTorch code van de Faster_RCNN implementatie.
- Het genereerde model moet aan de assets folder van het Android project worden toegevoegd.

Code/Detectiesystemen/Faster_RCNN/TensorFlow:
- Bevat de TensorFlow code van de Faster_RCNN implementatie.
- Het genereerde model moet aan de assets folder van het Android project worden toegevoegd.

Code/Herkenningssystemen/Faster_RCNN/ONNX:
- Bevat de ONNX code van de Faster_RCNN implementatie.
- De ONNX modellen worden gegenereerd in de TensorFlow en PyTorch notebook van Faster_RCNN.
- Het genereerde model moet aan de res/raw folder van het Android project worden toegevoegd.

Code/Detectiesystemen/YOLO/PyTorch:
- Bevat de PyTorch code van de YOLO implementatie.

Code/Detectiesystemen/YOLO/TensorFlow:
- Bevat de TensorFlow code van de YOLO implementatie.

## Documenten folder
Onder de folder documenten zijn de volgende items terug te vinden:
- Masterproef_tekst
- Litratuurstudie
- Tussentijdse presentatie
- Titelblad
- Opstartverslag
- Abstract (EN)
- Abstract (NL)
- Samenvatting
