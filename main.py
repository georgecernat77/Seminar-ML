from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# incarcarea setului de date
digits = datasets.load_digits()

#EX1

# se creeaza datele de antrenament si testare
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)

# se intitiliazieaza adaboost
clf = AdaBoostClassifier(n_estimators=100)

# antrenam modelul
clf.fit(X_train, y_train)

#functia care se ocupa de prezicerea cifrei
def predict_digit(image_path):
    # transformam imaginea intr un vector
    img = Image.open(image_path).convert('L')
    img = img.resize((8, 8), Image.ANTIALIAS)
    img_data = np.array(img).reshape(1, -1)

    # apelam predict pentru a prezice cifra
    digit = clf.predict(img_data)[0]
    return digit

# testam modelul pe setul de date de testare
y_pred = clf.predict(X_test)


# apelarea functiei
image_path = "imagine.png"
print(f'Cifra este: {predict_digit(image_path)}')

#EX2
#aplicam 10-means folosind setul de date de la exercitiul anterior
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(digits.data)

# afisam centroizi
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
plt.show()
