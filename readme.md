# KALI-ROULETTE : Générateur d'Entropie Vidéo

**Kali-Roulette** est une application de simulation de Casino (Roulette Européenne) dont le cœur repose sur un Générateur de Nombres Pseudo-Aléatoires (PRNG) conçu sur mesure.

Ce projet génère de l'aléa cryptographique en extrayant l'entropie d'un flux vidéo chaotique combiné à l'horloge système de haute précision.

---

## Fonctionnalités Clés

* **Moteur d'Entropie Hybride** : Fusion de données spatiales (pixels vidéo) et temporelles (nanosecondes).
* **Algorithme de Blanchiment** : Utilisation de **SHA-512** pour garantir une distribution uniforme et imprédictible.
* **Interface Moderne (GUI)** : Application développée avec **PyQt6** (Thème Dark, Responsive).
* **Architecture Asynchrone** : Utilisation de `QThread` pour la lecture vidéo et les calculs lourds, garantissant une interface fluide.
* **Module d'Analyse Statistique** : Outils de Data Science intégrés (**Matplotlib**) pour valider la qualité de l'aléa (Loi des Grands Nombres, Histogrammes).

---

## Installation et Démarrage

### Pré-requis
* Python 3.8 ou supérieur.

### Dépendances
```bash
pip install opencv-python PyQt6 matplotlib numpy
```

---

## Choix Techniques & Algorithmiques

Ce projet a été conçu pour répondre à une problématique de sécurité et d'équité des jeux de hasard.

### 1. La Source d'Entropie (Pourquoi Vidéo + Temps ?)
* **Vidéo seule (`kalicasino.mp4`)** : Bien que chaotique visuellement, un fichier vidéo est déterministe. Les pixels restent identiques à chaque lecture.
* **Temps seul (`time.time_ns()`)** : Partiellement prédictible si l'attaquant connaît l'instant exact de l'exécution.
* **La Solution (Salage)** : En mélangeant les octets de l'image actuelle (Crop central) avec le timestamp en nanosecondes, nous obtenons une graine (**seed**) unique et impossible à reproduire.

### 2. Le "Blanchiment" via SHA-512
Les images brutes présentent des biais (ex: zones sombres). Pour corriger cela, nous passons les données dans une fonction de hachage cryptographique **SHA-512**. 
* **Effet d'avalanche** : Une modification d'un seul pixel ou d'une nanoseconde change totalement le hash de sortie.

### 3. Projection vs Modulo (Mathématiques)
Pour transformer notre hash en un nombre entre 0 et 36, nous évitons le modulo (`% 37`) qui peut introduire un biais statistique (principe des tiroirs).
* **Notre Solution** : Conversion du hash en un flottant entre 0.0 et 1.0, puis projection sur le segment [0, 37[.
* **Formule** : $Resultat = \lfloor Float\_Cryptographique \times 37 \rfloor$

### 4. Architecture Logicielle
* **PyQt6** : Choix porté sur la robustesse du threading natif (`QThread`) et le rendu professionnel via feuilles de style (QSS).
* **Batch Processing** : Lors de l'analyse de 100 000 tirages, le calcul est segmenté pour permettre la mise à jour de la barre de progression sans geler l'UI.

---

## Validation Statistique

L'application intègre un module de preuve permettant de générer jusqu'à 1 000 000 de nombres pour observer :
1.  **L'Histogramme de Répartition** : Fréquence de sortie de chaque numéro (0-36).
2.  **La Courbe de Régularité** : Visualisation de la convergence vers la moyenne théorique (Loi des Grands Nombres).

---

## Structure du Projet

```text
kali-roulette/
│
├── main.py              # Point d'entrée unique (Logique + UI)
├── kalicasino.mp4       # Source d'entropie vidéo
├── README.md            # Documentation
└── requirements.txt     # Liste des bibliothèques
```
