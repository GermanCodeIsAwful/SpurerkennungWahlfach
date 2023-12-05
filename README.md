# SpurerkennungWahlfach
Bei diesem Projekt handelt es sich um eine Prüfungsleistung für den Kurs Digitale Bildverarbeitung an der 
Dualen Hochschule Baden-Württemberg. 
Weitere Informationen zu den Anforderungen finden Sie unter [Aufgabe.pdf](aufgabe.pdf).

![Test-Gif](/doc/vid/vid_for_readme_md.gif)

## Ordnerstruktur

```Text
|- CarSim_Build_v1: verschiedene Konfigurationen
    |  CarSim.exe: Starten CarSim
    
|- CarSim_Projekt_[...]: Verbindung und verarbeitung vom CarSim
    |  PythonServer_TCP_student.py: Start Lande detection für CarSim
    
|- config: verschiedene Konfigurationen

|- doc: Projekt-Dokumentation 
    |- class_description: Beschreibung des Quellcodes der erstellten Klassen
    |- vid: Aufnahme der verschiedenen Aufgaben
    
|- img: Bilder und Videos zum Projekt
    |- KITTI
    |- UDACITY
    
|- src: Feature Backbone
|  main.py: Start der 3 Videos
```

## Grobe Vorgehensweise

Als wir uns mit dieser Aufgabe beschäftigt hatten, haben wir eine wissenschaftliche Recherche durchgeführt.
Dabei sind wir unter anderem auf den Artikel "An Efficient Lane Line Detection Method Based on Computer Vision" (Quelle: 1).
Wir haben uns für den Sliding Window Ansatz entschieden, anstatt der in der Vorlesung vorgestellten Methoden,
da dieser Ansatz vielversprechender bei der Variation der Bildeinstellung ist.

1. Größenänderung und Transformieren
2. Sliding Window
   1. Segmentierung in Region of Interest
   2. Thresholding in HSV und Graustufe
   3. Histogramm erstellen
   4. Fenstern zeichen
      - Start beim Maximalwert des Histogramms
      - danach über den Mittelwert der Auswahl iterieren
   5. Polyfit mit Filter
   6. Rücktransformation
3. Visualisierung

→ Eine genaue Erklärung der einzelnen Schritte finden sie unter [Dokumentation](doc/class_description).

## Bonusaufgabe

Wir haben uns für folgende Bonusaufgaben entschieden:

- challenge_video  
- CarSim  
- Objekte im Bild


- (harder_challenge_video)

## Diskussion der Ergebnisse

Die [Ergebnisse](doc/vid) werden im folgenden Kapitel erörtert.

### Viedeo (99.999%) 

Das Hauptvideo der Aufgabe wird bis auf ein oder zwei Frames vollständig erkannt
und das Polynom wird korrekt berechnet.
Bei dieser Aufgabe wurden keine Straßenbahnprobleme gefunden, 
auch die hellen Bereiche werden im Thershold herausgefiltert.

### Challenge Viedeo (98%)

Das zweite Video wurde bis auf eine Stelle vollständig erkannt. 
Es gibt einen Bereich relativ am Anfang, wo unter einer Brücke gefahren wird. 
In dem dunklen Bereich gibt es fast keinen Unterschied mehr zwischen der Fahrbahn und der Markierung, 
da die automatische iso-Filterung nicht direkt hinterherkommt. 
Bei einer längeren Strecke könnte dies besser herausgefiltert werden. 
Auch die "Blendung" der Kamera nach der Brücke kann zu einem Problem führen.  

Zusätzlich befindet sich in der Mitte der Fahrbahn eine weiße Raute, die im Histogramm zu erkennen wird. 
Diese werden im vorliegenden Ansatz herausgefiltert.

### Harder Challenge Viedeo (60%)

Das letzte Viedo wird nur zum Teil erkannt. Mit dem derzeitigen Ansatz sind mehrere Probleme vorhanden:

- Das Bild schwenkt zu stark in eine Richtung, so dass die Fahrspur auf der anderen Seite nicht erfasst wird.
- Es gibt extrem viele Wechsel zwischen Sonneneinstrahlung und Schatten.  
  → Dynamische Anpassung des Therschold.
- Der Rasen und die weiße Markierung haben oft den gleichen Weißwert.  
  → Dynamische Anpassung des ROI.
- Blendung in der Windschutzscheibe
- Fehlende Markierung

Zusammenfassend kann gesagt werden, dass die gelbe Markierung durch den HSV-Raum deutlich besser erkannt wird als die 
weiße Markierung durch eine Grauabstufung.

### Simulation (100%)

Die Simulation läuft bis zu einer Geschwindigkeit von ca. 100-110 km/h problemlos. 
Nur an einer Stelle schlingert das Auto leicht in der Spur.

## Fazit

Das Programm läuft mit mehr als 20 FPS selbst im Debug Modus, es wurde unter anderem ein [CallGraph](doc/CallGraphV1.png) betrachtet. 
Mit Hilfe eines Profilers wurden bestimmte Codeabschnitte in der Laufzeit erkannt und verbessert.
Als mögliche Schritte wurden umgesetzt: Image Down Sizing, vollständige Initialisierung und Funktionsoptimierung.
Auf den konkreten Fall wurde in den jeweiligen [Dokumentationen](doc/class_description) näher eingegangen.

Als Lessons learned würden wir den Punkt mit der Objekterkennung betrachten. 
Wir sind an diesem Punkt anfänglich falsch angegangen.
Allerdings wurden richtige Methoden und Netze durch Recherche gefunden und angewendet.

## Ausblick

Um die Leistung und Effizienz des Programmes zu verbessern, haben wir folgende Möglichkeiten in Betracht gezogen:

- Optimierung von Datenstrukturen: 
  - Verwendung optimierter Datenstrukturen, die speziell für Sliding Windows geeignet sind, und Optimierung des Zugriffs auf Elemente im Fenster, um die Leistung zu verbessern.

- Unnötige Berechnungen vermeiden: 
  - Minimierung wiederholter oder unnötiger Berechnungen durch Speicherung oder Vorberechnung von Zwischenergebnissen.

- Parallelisierung oder Multithreading: 
  - Wo möglich und sinnvoll, können parallele Verarbeitungstechniken eingesetzt werden, um die Verarbeitungszeit des Sliding Window zu reduzieren.

- Weiteres Profiling und Analyse: 
  - Eine detaillierte Analyse des Codes durchführen, um Engpässe und ineffiziente Stellen zu identifizieren. Anschließend können gezielte Optimierungen vorgenommen werden.

Es können auch andere Methoden / Algorithmen für die Linienmarkierung verwendet werden. 
Sliding Window wird jedoch als empirisch sehr gut angesehen.

## Quellen

[1:iopscience](https://iopscience.iop.org/article/10.1088/1742-6596/1802/3/032006/pdf)

Weitere Codequellen sind meist direkt im Code angegeben.