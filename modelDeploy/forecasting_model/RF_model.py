import joblib

class ForcastModel:
    def __init__(self):
        self.forcast_model = None
    
    def predict(self, X:list[list[float]]):
        """Input the characteristic parameters of the model in GPU, and the output delay increases the prediction result

        Args:
            X (list[list[float]]): 
            The first list inputs the characteristics of the test model 
            [model L2 cache utilization, model GFLOPS, resource allocation of the model], 
            and the next list inputs the parameters of the interference model
        Returns:
            float: delay increases the prediction result
        """
        assert self.forcast_model is not None, "Model not loaded"
        coefficients=[1.98682567e-06,1.28970058e-05]
        
        L2_GFLOPS_sum=0
        inter_model_use=sum(row[2] for row in X[1:])
        
        for i in range(1,len(X)):
            L2_GFLOPS_sum+=X[i][0]*X[i][1]*(X[i][2]/(inter_model_use+1e-6))
        
        feature1=coefficients[0]*X[0][0]*X[0][1]+coefficients[1]*L2_GFLOPS_sum
        feature2=X[0][2]
        feature3=inter_model_use
        feature4=len(X)-1
        if feature4==0:
            return 0
        return self.forcast_model.predict([[feature1,feature2,feature3,feature4]])[0]

    
    def load_model(self,filename):
        self.forcast_model = joblib.load(filename)


if __name__=="__main__":
    model=ForcastModel()
    model.load_model("modelDeploy/forecasting_model/random_forest_model.pkl")
    X=[[70,1000,20]]
    print(model.predict(X))
