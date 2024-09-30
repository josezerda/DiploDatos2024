#!/usr/bin/python3.10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm.notebook import tqdm

import time

################################################################################
#                       Preprocesamiento
################################################################################

def preproc(DataFrame, cols_binary, cols_non_binary, BATCH_SIZE):
    '''
    Creamos los tensores de features y target, escalando los datos entre 0 y 1 
    para las variables no binarias. Luego, creamos un TensorDataset a partir de 
    los tensores de features y target. Finalmente, dividimos los datos para 
    entrenar, validar y testear, y creamos los cargadores de datos para leer 
    los datos por mini-batch.

    Args:
        DataFrame: conjunto de datos a procesar.
        cols_binary: ID de variables binarias.
        cols_non_binary: ID de variables no binarias.
        BATCH_SIZE: tamaño de lote.
    '''

    scaler = MinMaxScaler()

    features = torch.tensor(scaler.fit_transform(np.array(DataFrame[cols_binary+cols_non_binary]))).float()
    target = torch.tensor(np.array(DataFrame['Diabetes_binary'])).float()

    Data = TensorDataset(features, target)

    test_split = int(len(target)*0.8)
    validate_split = int(len(target)*0.1)
    train_split = len(target) - test_split - validate_split

    Data_train, Data_val, Data_test = random_split(Data, [train_split, validate_split, test_split])

    Load_train = DataLoader(Data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    Load_val = DataLoader(Data_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    Load_test = DataLoader(Data_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return Data_train, Data_val, Data_test, Load_train, Load_val, Load_test

################################################################################
#                       Entrenamiento y validación
################################################################################

def train(model, trainloader, loss_function, optimizer, epoch, device, use_tqdm=True):
    '''
    Lleva a cabo el entrenamiento del modelo.

    Args:
        model: estructura de la red neuronal.
        trainloader: cargador de datos de entrenamiento.
        loss_function: función de costo a utilizar.
        optimizer: tipo de descenso por gradiente a utilizar.
        epoch: número de épocas a entrenar.
        use_tqdm: muestra el progreso del entrenamiento.
        device: dónde se realiza el cálculo.
    '''

    # Enviamos el modelo al dispositivo donde se realiza el cálculo
    model.to(device)

    # Activamos el modo de entrenamiento en el modelo
    model.train()

    # Inicializamos el costo acumulado de la época
    training_loss = 0.0
    pbar = tqdm(trainloader) if use_tqdm else trainloader
    for step, (inputs, labels) in enumerate(pbar, 1):
        # Tensors to gpu (if necessary)
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients to zero
        optimizer.zero_grad()
        # Run a forward pass
        predicted_outputs = model(inputs.view(inputs.shape[0], -1))
        # Compute loss
        loss = loss_function(predicted_outputs, labels.long())
        # Backpropagation
        # Compute gradients
        loss.backward()

        # Accumulate the average loss of the mini-batch
        training_loss += loss.item()
        # Update the parameters
        optimizer.step()

        # Print statistics each 50 mini-batches
        if use_tqdm and step % 50 == 0:
          # Show number of epoch, step and average loss
          pbar.set_description(f"[{epoch}, {step}] loss: {training_loss / step:.4g}")

    epoch_training_loss = round(training_loss / len(trainloader), 4)

    return epoch_training_loss

def validation(model, valloader, loss_function, device, use_tqdm=True):
    '''
    Lleva a cabo la validación del modelo. Se utiliza la accuracy como métrica 
    principal y el f1-score como métrica secundaria.

    Args:
        model: estructura de la red neuronal.
        valloader: cargador de datos de validación.
        use_tqdm: muestra el progreso del entrenamiento.
        device: dónde se realiza el cálculo.
    '''

    model.eval()  # Activate evaluation mode
    y_true = []
    y_pred = []
    validation_loss = 0.0
    running_accuracy = 0.0
    total = 0
    # Don't calculate gradient speed up the forward pass
    with torch.no_grad():
        pbar = tqdm(valloader) if use_tqdm else valloader
        for (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            # Run the forward pass
            predicted_outputs = model(inputs.view(inputs.shape[0], -1))
            # Compute loss
            loss = loss_function(predicted_outputs, labels.long())
            # Accumulate the average loss of the mini-batch
            validation_loss += loss.item()

            # The label with the highest value will be our prediction
            _, predicted = torch.max(predicted_outputs , 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    epoch_validation_loss = round(validation_loss / len(valloader), 4)

    # Calculate metrics
    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = round(metrics.f1_score(y_true, y_pred, average='macro'), 4)

    return epoch_validation_loss, (accuracy, f1)

################################################################################
#                       Correr y guardar experimentos
################################################################################

def run_experiment(model, n_epochs, trainloader, valloader, loss_function, optimizer, device, use_tqdm=True):
    '''
    Función de ejecución de experimentos, que entrena y valida el modelo, 
    evaluando la función de costo para cada conjunto de hiperparámetros. Guarda 
    los hiperparámetros y resultados en un diccionario.

    Args:
        model: estructura de la red neuronal.
        n_epochs: número de épocas a entrenar.
        trainloader: cargador de datos de entrenamiento.
        valloader: cargador de datos de validación.
        loss_function: función de costo a utilizar.
        optimizer: tipo de descenso por gradiente a utilizar.
        device: dónde se realiza el cálculo.
        use_tqdm: muestra el progreso del entrenamiento.
    '''

    register_performance = {
        'epoch': [],
        'epoch_training_loss': [], 'epoch_validation_loss': [],
        'validation_accuracy': [], 'validation_f1': []
        }
    best_accuracy = 0.0

    print("Begin training...")
    start = time.time()
    # Loop through the dataset multiple times
    for epoch in range(1, n_epochs + 1):
        # Train the model
        epoch_training_loss = train(model, trainloader, loss_function, optimizer, epoch, device, use_tqdm)
        # Validate the model
        epoch_validation_loss, metrics = validation(model, valloader, loss_function, device, use_tqdm)

        register_performance['epoch'].append(epoch)
        register_performance['epoch_training_loss'].append(epoch_training_loss)
        register_performance['epoch_validation_loss'].append(epoch_validation_loss)
        register_performance['validation_accuracy'].append(metrics[0])
        register_performance['validation_f1'].append(metrics[1])

        # Save the model if the accuracy is the best
        if metrics[0] > best_accuracy:
            best_model = model
            best_accuracy = metrics[0]

        if (epoch % 10 == 0) and (epoch != n_epochs):
            print(f'\tVoy por la época {epoch}! :)')
        elif epoch == n_epochs:
            WallTime = time.time() - start
            print(f'\tTerminé! :D >>> WallTime = {WallTime/60:.2f} min')


    # Save the results
    experiment = {
        'arquitecture': str(model),
        'optimizer': optimizer,
        'loss': str(loss_function),
        'epochs': n_epochs,
    }

    # Print the statistics of the epoch
    print(f'Completed training in {epoch} batch: ',
          'Training Loss is: ' , epoch_training_loss,
          '- Validation Loss is: ', epoch_validation_loss,
          '- Accuracy is: ', (metrics[0]),
          '- F1 is: ', (metrics[1])
          )
    return experiment, register_performance, best_model

def get_data_loss_metrics(experiments_set, path):
    df_base = pd.DataFrame()
    for i in range(len(experiments_set)):
        arquitecture = experiments_set[i][0]['arquitecture']
        model_name = arquitecture.split('(')[0]
        if len(arquitecture.split('activ1): ')) == 1:
            activation_function_name = arquitecture.split('(1): ')[1].split('()\n')[0]
        else:
            activation_function_name = arquitecture.split('activ1): ')[1].split('()\n  (drop1)')[0].split('(negative_slope')[0]
        optim = type(experiments_set[i][0]['optimizer']).__name__
        lr = experiments_set[i][0]['optimizer'].param_groups[0]['lr']
        weight_decay = experiments_set[i][0]['optimizer'].param_groups[0]['weight_decay']
        df = pd.DataFrame(experiments_set[i][1])
        df['model-activation-optimizer-lr-wd'] = f'{model_name}-{activation_function_name}-{optim}-{lr}-{weight_decay}'
        df_base = pd.concat([df_base, df])

    df_base.to_csv(path, index=False)

    df_metrics = df_base.drop(columns=['epoch_training_loss', 'epoch_validation_loss'])
    df_loss = df_base.drop(columns=['validation_accuracy', 'validation_f1']).melt(id_vars=['epoch', 'model-activation-optimizer-lr-wd'],
                                                                                        value_vars=['epoch_training_loss', 'epoch_validation_loss'],
                                                                                        var_name='task', value_name='loss')
    return df_loss, df_metrics

################################################################################
#                       Graficar
################################################################################

def plot_results(n_epochs, path, experiments_set=None):

    L = [k*10+9 for k in range(int(n_epochs/10))]

    if experiments_set == None:
        df_base = pd.read_csv(path)
        df_metrics = df_base.drop(columns=['epoch_training_loss', 'epoch_validation_loss'])
        df_loss = df_base.drop(columns=['validation_accuracy', 'validation_f1']).melt(id_vars=['epoch', 'model-activation-optimizer-lr-wd'],
                                                                                            value_vars=['epoch_training_loss', 'epoch_validation_loss'],
                                                                                            var_name='task', value_name='loss')
    else:
        df_loss, df_metrics = get_data_loss_metrics(experiments_set, path)

    print('Pérdidas:')
    sns.catplot(data=df_loss, x='epoch', y='loss',  hue='task', col='model-activation-optimizer-lr-wd',
                col_wrap=3, kind='point', height=4, aspect=1.5)
    plt.xticks(L)
    plt.show()

    print('\nMétricas:')
    _, axs = plt.subplots(1, 2, figsize=(15, 4))
    sns.pointplot(data=df_metrics, x='epoch', y='validation_accuracy', hue='model-activation-optimizer-lr-wd', ax=axs[0])
    axs[0].set_xticks(L)
    sns.pointplot(data=df_metrics, x='epoch', y='validation_f1', hue='model-activation-optimizer-lr-wd', ax=axs[1])
    axs[1].set_xticks(L)
    plt.show()

################################################################################
#                       Correr, guardar y graficar el mejor caso
################################################################################

def run_best(model, n_epochs, trainloader, valloader, testloader, loss_function, optimizer, device, use_tqdm=True):
    '''
    Ejecuta la corrida con los mejores hiperparámetros encontrados, entrenando, 
    validando y evaluando el modelo.

    Args:
        model: estructura de la red neuronal.
        n_epochs: número de épocas a entrenar.
        trainloader: cargador de datos de entrenamiento.
        valloader: cargador de datos de validación.
        testloader: cargador de datos de evaluación.
        loss_function: función de costo a utilizar.
        optimizer: tipo de descenso por gradiente a utilizar.
        device: dónde se realiza el cálculo.
        use_tqdm: muestra el progreso del entrenamiento.
    '''

    register_performance = {
        'epoch': [],
        'epoch_training_loss': [], 
        'epoch_validation_loss': [], 
        'epoch_testing_loss': [], 
        'training_accuracy': [], 'training_f1': [],
        'validation_accuracy': [], 'validation_f1': [],
        'testing_accuracy': [], 'testing_f1': []
        }
    best_accuracy = 0.0

    print("Begin training...")
    start = time.time()
    # Loop through the dataset multiple times
    for epoch in range(1, n_epochs + 1):
        register_performance['epoch'].append(epoch)
        # Train the model
        epoch_training_loss = train(model, trainloader, loss_function, optimizer, epoch, device, use_tqdm)
        register_performance['epoch_training_loss'].append(epoch_training_loss)
        _, metrics_train = validation(model, trainloader, loss_function, device, use_tqdm)
        register_performance['training_accuracy'].append(metrics_train[0])
        register_performance['training_f1'].append(metrics_train[1])

        # Validate the model
        epoch_validation_loss, metrics_val = validation(model, valloader, loss_function, device, use_tqdm)
        register_performance['epoch_validation_loss'].append(epoch_validation_loss)
        register_performance['validation_accuracy'].append(metrics_val[0])
        register_performance['validation_f1'].append(metrics_val[1])
        
        # Test the model
        epoch_testing_loss, metrics_test = validation(model, testloader, loss_function, device, use_tqdm)
        register_performance['epoch_testing_loss'].append(epoch_testing_loss)
        register_performance['testing_accuracy'].append(metrics_test[0])
        register_performance['testing_f1'].append(metrics_test[1])

        # Save the model if the accuracy is the best
        if metrics_val[0] > best_accuracy:
            best_model = model
            best_accuracy = metrics_val[0]

        if (epoch % 10 == 0) and (epoch != n_epochs):
            print(f'\tVoy por la época {epoch}! :)')
        elif epoch == n_epochs:
            WallTime = time.time() - start
            print(f'\tTerminé! :D >>> WallTime = {WallTime/60:.2f} min')

    # Save the results
    experiment = {
        'arquitecture': str(model),
        'optimizer': optimizer,
        'loss': str(loss_function),
        'epochs': n_epochs,
    }

    # Print the statistics of the epoch
    print(f'Completed training in {epoch} batch: ',
          'Training Loss is: ' , epoch_training_loss,
          'Training Accuracy is: ', (metrics_train[0]),
          'Training F1 is: ', (metrics_train[1]),
          'Validation Loss is: ', epoch_validation_loss,
          'Validation Accuracy is: ', (metrics_val[0]),
          'Validation F1 is: ', (metrics_val[1]),
          'Testing Loss is: ', epoch_testing_loss,
          'Testing Accuracy is: ', (metrics_test[0]),
          'Testing F1 is: ', (metrics_test[1])
          )
    return experiment, register_performance, best_model

def get_best_loss_metrics(best_exp, path):
    df_base = pd.DataFrame()
    
    arquitecture = best_exp[0]['arquitecture']
    model_name = arquitecture.split('(')[0]
    activation_function_name = arquitecture.split('activ1): ')[1].split('()\n  (drop1)')[0].split('(negative_slope')[0]
    optim = type(best_exp[0]['optimizer']).__name__
    lr = best_exp[0]['optimizer'].param_groups[0]['lr']
    weight_decay = best_exp[0]['optimizer'].param_groups[0]['weight_decay']
    df = pd.DataFrame(best_exp[1])
    df['model-activation-optimizer-lr-wd'] = f'{model_name}-{activation_function_name}-{optim}-{lr}-{weight_decay}'
    df_base = pd.concat([df_base, df])

    df_base.to_csv(path, index=False)

    df_metrics = df_base.drop(columns=['epoch_training_loss', 'epoch_validation_loss', 'epoch_testing_loss'])
    df_loss = df_base.drop(columns=['training_accuracy', 'training_f1', 'validation_accuracy', 'validation_f1', 'testing_accuracy', 'testing_f1']).melt(id_vars=['epoch', 'model-activation-optimizer-lr-wd'],
                                                                                        value_vars=['epoch_training_loss', 'epoch_validation_loss', 'epoch_testing_loss'],
                                                                                        var_name='task', value_name='loss')
    return df_loss, df_metrics

def plot_best(n_epochs, path, experiments_set=None):

    L = [k*10+9 for k in range(int(n_epochs/10))]

    if experiments_set == None:
        df_base = pd.read_csv(path)
        df_metrics = df_base.drop(columns=['epoch_training_loss', 'epoch_validation_loss', 'epoch_testing_loss'])
        df_loss = df_base.drop(columns=['training_accuracy', 'training_f1', 'validation_accuracy', 'validation_f1', 'testing_accuracy', 'testing_f1']).melt(id_vars=['epoch', 'model-activation-optimizer-lr-wd'],
                                                                                            value_vars=['epoch_training_loss', 'epoch_validation_loss', 'epoch_testing_loss'],
                                                                                            var_name='task', value_name='loss')
    else:
        df_loss, df_metrics = get_best_loss_metrics(experiments_set, path)

    print('Pérdidas:')
    sns.catplot(data=df_loss, x='epoch', y='loss',  hue='task', col='model-activation-optimizer-lr-wd',
                col_wrap=1, kind='point', height=4, aspect=1.5)
    plt.xticks(L)
    plt.show()

    print('\nMétricas:')
    sns.pointplot(data=df_metrics, x='epoch', y='training_accuracy', color='C0', label='Train')
    sns.pointplot(data=df_metrics, x='epoch', y='validation_accuracy', color='C1', label='Val')
    sns.pointplot(data=df_metrics, x='epoch', y='testing_accuracy', color='C2', label='Test')
    plt.ylabel('Accuracy')
    plt.xticks(L)
    plt.legend(loc='lower right')
    plt.show()