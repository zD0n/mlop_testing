import data_validation,data_preprocessing,train_evaluate_register,transition_model

model_name= "Emotion-Classifier"
epoch= 10

data_validation.validate_data()

prepro = data_preprocessing.preprocess_data()

check=False
threshold = 0.8

while check != True:

    newthreshold = train_evaluate_register.train_evaluate_register(preprocessing_run_id=prepro,epochs=epoch,model_name=model_name,ACCURACY_THRESHOLD=threshold)

    check = transition_model.transition_model_alias(model_name,"Staging")

    threshold = threshold
    epoch+=5
