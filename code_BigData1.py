import numpy as np
from time import clock

DonneTrn = np.load('E:\\dossieer\\cours_BigData\\chiffres\\data\\trn_img.npy')
DonneTrn_lbl = np.load('E:\\dossieer\\cours_BigData\\chiffres\\data\\trn_lbl.npy')
DonneDev = np.load('E:\\dossieer\\cours_BigData\\chiffres\\data\\dev_img.npy')
DonneDev_lbl = np.load('E:\\dossieer\\cours_BigData\\chiffres\\data\\dev_lbl.npy')
DonneTst = np.load('E:\\dossieer\\cours_BigData\\chiffres\\data\\tst_img.npy')

##c'est un liste pour enregistrer tout les classe, ça veut dire 1,2,3,4,5,6,7,8,9,0
listClasseDmin = []
	
def trainDmin(trainMatrix,trainLabel):
	global listClasseDmin
	##c'est l'initialisation de la liste de classe. 
	##len(trainMatrix[0]), c'est pour prendre le nombre de variable qu'on a utiliser pour l'appretissage et prediction
	listClasseDmin = np.zeros(shape = (10,len(trainMatrix[0])))
	## pour prendre le nombre de chaque classe
	listNum = np.bincount(trainLabel)
	for indice in range(len(trainMatrix)):
	## calculer le nombre total de chaque classe
		listClasseDmin[trainLabel[indice]] = listClasseDmin[trainLabel[indice]]+trainMatrix[indice]
	## calculer la moyenne, on peut perdre le le vecteur qui représente les classe 
	listClasseDmin = (listClasseDmin.T/listNum).T

def predictDmin(predictMatrix):
	listVariance = np.zeros(shape = (len(predictMatrix),10))
	## ici, on a bien fait vectorisatoin, on a choisi de parcourir liste de classe au lieu de matrice de la prédiction, parce qu'on a juste 10 classe, mais on a 5000 lignes de matrice
	for indice in range(len(listClasseDmin)):
		## listVariance[:,indice] représente les distances entre tout les vecteur de matrice de la prédiction et le classe listVariance[:,indice]
		listVariance[:,indice] = ((predictMatrix-listClasseDmin[indice])**2).sum(axis = 1)
	## prendre le distance minimal entre représentant d'un classe et un vecteur 
	return np.argmin(listVariance,axis = 1)

def calTauxErreur(listPredict,listLavel):
	## pour obtenir le nombre qui est bien predit
	nombreCorrect = np.sum(np.array(listPredict) == np.array(listLavel))
	print("Taux d'erreur est:",(len(listLavel)-nombreCorrect)/(len(listLavel)))

def exo1():
	##Dmin sans acp
	print("Dmin sans acp")
	trainDmin(DonneTrn,DonneTrn_lbl)
	preList = predictDmin(DonneDev)
	calTauxErreur(preList,DonneDev_lbl)	
	##avec acp
	print("Dmin avec acp=10")
	## si on veut utiliser 20 points, ce qu'on doit faire c'est juste de changer le valeur 10 pour 20
	p = acpCompress(DonneTrn,10)
	## pour prendre le matrice qui compressé par l'algorithme ACP
	listClassKNN = np.matmul(DonneTrn,p)
	predictMatrix = np.matmul(DonneDev,p)	
	trainDmin(listClassKNN,DonneTrn_lbl)
	preList = predictDmin(predictMatrix)
	calTauxErreur(preList,DonneDev_lbl)	


def acpCompress(matrice,nombreChoisit):
	
	C = np.cov(matrice,rowvar=0)
    # v : tableau des valeurs propres
    # w : matrice des vecteurs propres
	v,w = np.linalg.eigh(C)
	p = w[:,len(w)-nombreChoisit: len(w)]
	return p

def predictKNN(listClassKNN,predictMatrix,trainLabel):
	Diff = np.zeros(len(predictMatrix))
	## on suppose que le matrice de l'apprentissage est souvent plus grand, donc on choisit parcourir le matrice de la prédilection 
	for indice in range(len(predictMatrix)):
		## pour prendre le distance entre vecteur  predictMatrix[indice] et tout les classe(on a 10000 classe)
		varience = ((listClassKNN - predictMatrix[indice])**2).sum(axis = 1)
		argIndice = np.argmin(varience)
		Diff[indice] = trainLabel[argIndice]
	return Diff


def exo2():
	print("KNN execution avec ACP = 10")
	## si on veut utiliser 20 points, ce qu'on doit faire c'est juste de changer le valeur 10 pour 20
	p = acpCompress(DonneTrn,20)
	listClassKNN = np.matmul(DonneTrn,p)
	predictMatrix = np.matmul(DonneDev,p)
	prediList = predictKNN(listClassKNN,predictMatrix,DonneTrn_lbl)
	calTauxErreur(prediList,DonneDev_lbl)

print("On utilise le fichier trn_img.npy pour faire l'apprentissage")
print("On utilise le fichier dev_img.npy pour faire la prediction")
exo1()
exo2()