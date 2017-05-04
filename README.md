# LogisticRegression
Create input files; x.txt, y.txt, and newInstace.txt, where:

 -  x.txt is the training matrix of the form (values are floating points)
 
      1 a b c d ...
     
      1 e f g h ...
      
      1 i j k l ...

      
 - y.txt is the corresponding class label vector for the training data (values are ints, must be either 1 or -1 correspoding to T/ of the form:
 
    -1 1 1 1 -1 -1 ....
 
 - newInstace.txt is the unclassified input vector (values are floating points)
 
    1 a b c d ...


Compile using:

  javac LogReg.java

Run:

  java LogReg

