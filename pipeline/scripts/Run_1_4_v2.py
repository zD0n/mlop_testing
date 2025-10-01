import data_validation,data_preprocessing,train_evaluate_register,transition_model

model_name= "Emotion-Classifier"
epoch= 1

data_validation.validate_data()

prepro = data_preprocessing.preprocess_data()

check=False

while check != True:

    train_evaluate_register.train_evaluate_register(preprocessing_run_id=prepro,epochs=epoch,model_name=model_name)

    check = transition_model.transition_model_alias(model_name,"Staging")

    epoch+=10
