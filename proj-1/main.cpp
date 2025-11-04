#include <iostream>
#include <cmath>
#include <random>
using namespace std;


double sigmoid(double z) {
    return 1.0/(1.0+exp(-z));
}

int main() {
    pair<int, int> x[10] = {{1, 1}, {1, 3}, {2, 5}, {2, 2}, {3, 4}, {4, 1}, {5, 2}, {6, 3}, {7, 5}, {8, 2}};
    int y[10] = {0, 0, 1, 0, 1, 0, 1, 1, 1, 1};

    double w1 = 0.2;
    double w2 = 0.2;
    double b = 0.2;
    double eta = 0.1;
    double gradw1;
    double gradw2;
    double gradb;
    double z;
    double yhat;
    double loss;
    double error;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, 9);
    for (int epoch = 0; epoch < 1000; epoch++) {
        int i = distrib(gen);
        int x1 = x[i].first, x2 = x[i].second;
        z = w1*x1 + w2*x2 + b;
        cout << '\n' << "FUNC: " << '\n' << w1 << "*x1" << "+" << w2 << "*x2" << "+" << b;
        yhat = sigmoid(z);
        loss = 0.5*(pow((yhat - y[i]), 2));
        gradw1 = (yhat - y[i]) * ((yhat)*(1-yhat)) * x1;
        gradw2 = (yhat - y[i]) * ((yhat)*(1-yhat)) * x2;
        gradb = (yhat - y[i]) * ((yhat)*(1-yhat));
        w1 -= eta * gradw1;
        w2 -= eta * gradw2;
        b -= eta * gradb;
        error = y[i] - yhat;
         
        cout << "YHAT: " << yhat << " ERROR: " << error << " X1: " << x1 << " X2: " << x2 << " W1F: " << w1 << " W2F: " << w2 << " W1 CHANGE: " << eta*gradw1 << " W2 CHANGE: " << eta*gradw2 << " BF: " << b << " B CHANGE: " << eta*gradb << " MSE LOSS: " << loss << endl;
    }
    cout << '\n' << "FUNC: " << '\n' << w1 << "*x1" << "+" << w2 << "*x2" << "+" << b;
}