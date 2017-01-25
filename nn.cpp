// This code begins from the base file written by Steven C. Shaffer.(see below)
// I will make edits to change this to a different type of gate, and/or to change the activation 
// Function etc. Akash Maharaj. Jan 2017
// Some edits made:
// 1. Added an OR gate (easy)  (01/24/2017)
// 2. Added a ReLU activation function - 
//    turns out that this is a bad idea for small networks like this on(01/24/2017)
// 3. TODO: add a leaky ReLU?
// 4. TODO: modify this neural network to fit a function!


// Copyright, 2015, by Steven C. Shaffer 
// You may use this code as a basis for your own learning or as the starting 
// place for your own projects (academic or commercial). Do not reproduce // or publish in whole or in part without the express permission of the 
// author. 


#include <stdio.h> 
#include <stdlib.h> 
#include <cmath> 


// Constants 
const int NUMINPUTNODES = 2; 
const int NUMHIDDENNODES = 2; 
const int NUMOUTPUTNODES = 1; 
const int NUMNODES = NUMINPUTNODES + NUMHIDDENNODES + NUMOUTPUTNODES; 
const int ARRAYSIZE = NUMNODES + 1; // 1-offset to match “node 1” “node
const int MAXITERATIONS = 131072;
const double E=2.71828;
const double LEARNINGRATE=0.3;

// Function prototypes
void initialize( double[][ ARRAYSIZE], double[], double[], double[]); 
void connectNodes( double[][ ARRAYSIZE], double[]); 
void trainingExampleXOR( double[], double[]); 
void trainingExampleOR( double[], double[]); 
void forwardPass( double[][ ARRAYSIZE], double[], double[]); 
double backwardPass( double[][ ARRAYSIZE], double[], double[], double[]); 
double sigmoidForward(double);
double reluForward(double);
double sigmoidBackward(double);
double reluBackward(double);
void displayNetwork( double[], double);


int main(){
	printf("Neural Network Program\n");

	double weights[ARRAYSIZE][ARRAYSIZE];
	double values[ARRAYSIZE];
	double expectedvalues[ARRAYSIZE];
	double thresholds[ARRAYSIZE];

	initialize(weights, values, expectedvalues, thresholds);
	connectNodes(weights,thresholds);

	int counter = 0;
	while(counter < MAXITERATIONS)
	{	
		trainingExampleXOR(values, expectedvalues);
		forwardPass(weights,values,thresholds);
		double sumOfSquaredErrors = backwardPass(weights, values, expectedvalues, thresholds);
		displayNetwork(values,sumOfSquaredErrors);
		counter++;
	}

	return 0;

}


//Initialize all the weights and biases in the network to be zero
void initialize(double weights[][ARRAYSIZE], double values[], 
	double expectedvalues[], double thresholds[]){

	for (int x = 0; x <= NUMNODES; x++)
	{
		values[x] = 0.0;
		expectedvalues[x]=0.0;
		thresholds[x] = 0.0;
		for(int y=0; y<= NUMNODES; y++)
		{
			weights[x][y]=0.0;
		}
	}
}

//Make all the connections between nodes
void connectNodes(double weights[][ARRAYSIZE], double thresholds[])
{
	for( int x = 1; x <= NUMNODES; x++){
		for( int y = 1; y <= NUMNODES; y++)
		{
			weights[x][y] = (rand()%200)/100.0; 		
		}

	}

	thresholds[3] = rand()/(double)rand();
	thresholds[4] = rand()/(double)rand();
	thresholds[5] = rand()/(double)rand();

	printf("%f%f%f%f%f%f\n%f%f%f\n",
		weights[1][3],weights[1][4],weights[2][3],
		weights[2][4],weights[3][5],weights[4][5],
		thresholds[3],thresholds[4],thresholds[5]
		);

}


void trainingExampleXOR( double values[], double expectedvalues[])
{
	static int counter = 0;

	switch (counter%4)
	{
	case 0:
		values[1] = 1;
		values[2] = 1;
		expectedvalues[5] = 0;
		break;
	case 1:
		values[1] = 0;
		values[2] = 1;
		expectedvalues[5] = 1;
		break;
	case 2:
		values[1] = 1;
		values[2] = 0;
		expectedvalues[5] = 1;
		break;
	case 3:
		values[1] = 0;
		values[2] = 0;
		expectedvalues[5] = 0;
		break;
	}
	counter++;
}



void trainingExampleOR( double values[], double expectedvalues[])
{
	static int counter = 0;

	switch (counter%4)
	{
	case 0:
		values[1] = 1;
		values[2] = 1;
		expectedvalues[5] = 1;
		break;
	case 1:
		values[1] = 0;
		values[2] = 1;
		expectedvalues[5] = 1;
		break;
	case 2:
		values[1] = 1;
		values[2] = 0;
		expectedvalues[5] = 1;
		break;
	case 3:
		values[1] = 0;
		values[2] = 0;
		expectedvalues[5] = 0;
		break;
	}
	counter++;
}



void forwardPass( double weights[][ARRAYSIZE], double values[], 
	double thresholds[])
{
	// for each hidden node (i.e first layer)
	for(int h = 1+ NUMINPUTNODES; h < 1+ NUMINPUTNODES+NUMHIDDENNODES; h++)
	{
		double weightedInput = 0.0;
		//add up the weighted input
		for(int i=1; i < 1+ NUMINPUTNODES; i++)
		{
			weightedInput += weights[i][h]*values[i];
		}
		// handle the thresholds
		weightedInput += thresholds[h];

		// Now pass through the Softmax/ Fermi function
		//values[h] = 1.0/(1.0 + pow(E,-weightedInput));
		values[h] = sigmoidForward(weightedInput);
	}

	//for each output node (i.e. second/ output layer)
	for (int o = 1+NUMINPUTNODES+NUMHIDDENNODES; o < 1+ NUMNODES; o++)
	{
		double weightedInput=0.0;
		for(int h=1 + NUMINPUTNODES; h < 1+ NUMINPUTNODES+NUMHIDDENNODES; h++)
		{
			weightedInput += weights[h][o]*values[h];
		}
		//handle the thresholds
		weightedInput += thresholds[o];


		// Once again, pass through the Softmax function.
		//values[o] = 1.0/(1.0 + pow(E,-weightedInput));
		values[o] = sigmoidForward(weightedInput);

	}
}


double backwardPass(double weights[][ARRAYSIZE], double values[], 
	double expectedvalues[], double thresholds[])
{
	double sumOfSquaredErrors=0.0;

	for (int o= 1+ NUMINPUTNODES + NUMHIDDENNODES; o < 1+NUMNODES; o++)
	{
		double absoluteError = expectedvalues[o]- values[o];
		sumOfSquaredErrors += pow(absoluteError,2);
		
		// This is gradient of softmax function
		double outputErrorGradient = sigmoidBackward(values[o])*absoluteError;

		//update each weight from the hidden layer
		for (int h = 1+NUMINPUTNODES;
			h < 1 + NUMINPUTNODES + NUMHIDDENNODES; h++)
		{
			double delta = LEARNINGRATE*values[h]*outputErrorGradient;
			weights[h][o] += delta;
			
			// Once more, this is gradient of softmax function
			double hiddenErrorGradient = sigmoidBackward(values[h])*
										outputErrorGradient*weights[h][o];
			for (int i=1; i < 1 + NUMINPUTNODES; i++)
			{
				double delta = LEARNINGRATE*values[i]*hiddenErrorGradient;
				weights[i][h] += delta;
			}

			double thresholdDelta = LEARNINGRATE  * hiddenErrorGradient;
			thresholds[h] += thresholdDelta;
		}

		//update each weight for the theta(??)
		double delta = LEARNINGRATE * outputErrorGradient;
		thresholds[o] += delta;
	}
	return sumOfSquaredErrors;
}

void displayNetwork(double values[], double sumOfSquaredErrors)
{
	static int counter = 0;
	if((counter%4) == 0)
		printf("----------------------------------------------\n");
	printf("%8.4f|",values[1]);
	printf("%8.4f|",values[2]);
	printf("%8.4f|",values[5]);
	printf(" err.%8.5f\n", sumOfSquaredErrors);
	counter++;
}


double sigmoidForward(double value)
{
	return 1.0/(1.0 + pow(E,-value));
}

double sigmoidBackward(double value)
{
	return value*(1-value);
}



// --- These are relu activations, which turn out to be a BAD idea for a small network
double reluForward(double value)
{
	if ( value >= 0.0){
		return value;
	} else {
		return 0.0;
	}
}

double reluBackward(double value)
{
	if (value >= 0.0){
		return 1.0;
	} else {
		return 0.0;
	}
}
