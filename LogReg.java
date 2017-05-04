import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

/*
 * @author Alexander Steel, May 2017
 * 
 * Logistic Regression:
 * 
 * This program reads in a classified (binary T/F) dataset and a new, unclassified instance. Using Stochastic Gradient Descent, a machine learning model is built and the 
 * probability that the new instance is in either the FALSE or the TRUE class is calculated. 
 * 
 */

public class LogReg {

	private static final double KAPPA = 0.001; // Kappa is the stepping value, i.e. how much of a step we take in the direction of steepest slope when updating the weight vector
	private static final int ITERATIONS = 30; // Fixed value of iterations for updating weight vector

	public static void main(String[] args){

		double[][] trainingData = readX();

		int[] y = readY();

		double[] newInstance = readNewInstance();

		double[] w = new double[trainingData[0].length]; // Weight vector

		for(int i = 0; i < w.length; i++){ // Weight vector is initially the zero vector
			w[i] = 0;
		}

		double[] gradient = new double[w.length]; // New vector to store each iteration of the gradient vector

		for(int i = 0; i < ITERATIONS; i++){ 

			// At each iteration, the gradient vector is calculated, multiplied by the stepping value kappa, and subtracted from the weight vector

			gradient = computeGradient(trainingData,y,w);
			w = vectorMinusVector(w,gradient);

		}

		System.out.println("Probability class = 1 (TRUE) given new instance x: " + classifyNewInstance(w,newInstance));
		System.out.println("Probability class = -1 (FALSE) given new instance x: " + (1.0 - classifyNewInstance(w,newInstance)));

	}
	private static double[][] readX() {

		String matrix = "";

		try {
			Scanner s3 = new Scanner(new File("x.txt"));
			while (s3.hasNext()){
				matrix += s3.next();
				if (s3.hasNextLine())
					matrix+= "%";
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		String[] tmpMatrix = matrix.split("%");

		String tmpCounter = tmpMatrix[0];

		int columns = 1;

		for(int i = 0; i < tmpCounter.length(); i++){
			if (tmpCounter.charAt(i) == ',')
				columns++;

		}

		String[][] m = new String[tmpMatrix.length][columns];

		for(int i = 0; i < m.length; i++){
			for(int j = 0; j < m[0].length; j++){
				m[i][j] = "0";
			}
		
		}

		for(int i = 0; i < m.length; i++){
			m[i] = tmpMatrix[i].split(",");
		}

		double[][] x = new double[tmpMatrix.length][columns];

		for(int i = 0; i < m.length; i++){
			for(int j = 0; j < m[0].length; j++){

				x[i][j] = Double.parseDouble(m[i][j]);
			}
		
		}



		return x;
	}
	private static double[] readNewInstance() {

		String newInstance = "";

		try {
			Scanner s2 = new Scanner(new File("newInstace.txt"));

			while (s2.hasNext()){
				newInstance += s2.next();
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		String[] newI = newInstance.split(",");

		double[] newIn = new double[newI.length];

		for(int i = 0; i < newI.length; i++){
			newIn[i] = Double.parseDouble(newI[i]);
		}

		return newIn;

	}
	private static int[] readY() {

		String y = "";

		try {
			Scanner s1 = new Scanner(new File("y.txt"));

			while (s1.hasNext())
				y+= s1.next();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		String[] yVector = y.split(",");

		int[] y2 = new int[yVector.length];

		for(int i = 0; i < y2.length; i++){
			y2[i] = Integer.parseInt(yVector[i]);
		}

		return y2;

	}
	/*
	 * computeGradient: Method which computes and returns the gradient vector 
	 */
	private static double[] computeGradient(double[][] trainingData, int[] y, double[] w) {

		double oneOverN = -1 / (double)y.length;
		double denominator = 0;
		double[] xk = new double[w.length];
		double[] tempVector = new double[w.length];
		double[] v = new double[w.length];

		for(int i = 0; i < v.length; i++) // Set values of temporary vector v to zero initially
			v[i] = 0;

		for(int k = 0; k < y.length; k++){

			xk = updateXk(xk,trainingData,k);
			denominator = 1 + Math.exp(y[k]*dotProduct(w,xk));
			tempVector = scalarTimesVector(xk,y[k]);
			tempVector = scalarTimesVector(tempVector,1/denominator);
			v = vectorPlusVector(v,tempVector);

		}

		v = scalarTimesVector(v,oneOverN);

		return v;
	}

	private static void printArray(double[] xk) {
		for(int i = 0; i < xk.length; i++)
			System.out.print(xk[i] +" " );

		System.out.println();


	}


	/*
	 * vectorPlusVector: Method which adds two equal sized vectors and returns the result as an array
	 */
	private static double[] vectorPlusVector(double[] v, double[] tempVector) {

		double[] tmp = new double[v.length];

		for(int i = 0; i < tmp.length; i++)
			tmp[i] = v[i] + tempVector[i];

		return tmp;
	}

	/*
	 * scalarTimesVector: Method which adds multiplies a vector by a scalar
	 */
	private static double[] scalarTimesVector(double[] xk, double scalar) {

		for(int i = 0; i < xk.length; i++)
			xk[i] = xk[i]*scalar;

		return xk;
	}

	/*
	 * updateXk: Method which extracts a row out of a matrix and returns it as a vector
	 */
	private static double[] updateXk(double[] xk, double[][] trainingData, int k) {

		for(int i = 0; i < xk.length; i++){
			xk[i] = trainingData[k][i];
		}

		return xk;
	}
	/*
	 * vectorMinusVector: Method which first multiplies a vector by the stepping value kappa, and then subtracts the second vector from the first vector
	 */
	private static double[] vectorMinusVector(double[] w, double[] gradient) {

		// first multiply gradient vector by kappa

		for(int i = 0; i < gradient.length; i++)
			gradient[i] = KAPPA * gradient[i];

		// then subtract the gradient vector from w

		for (int i = 0; i < w.length; i++)
			w[i] = w[i] - gradient[i];

		return w;
	}
	/*
	 * classifyNewInstance: Method which computes the probability of a new instance being in class 1 (True)
	 * Uses the formula for the sigmoid function f(x,w) = 1 / (1 + e^(-w.x))
	 */
	private static double classifyNewInstance(double[] w, double[] newInstance) {

		return 1 / (1 + Math.exp(-1*dotProduct(w,newInstance)));
	}

	/*
	 * dotProduct: Method which computes the dot product of two equal sized vectors and returns the result as an array
	 */

	private static double dotProduct(double[] w, double[] newInstance) {

		double tmp = 0;

		for (int i = 0; i < w.length; i++)
			tmp += w[i]*newInstance[i];

		return tmp;			
	}

}
