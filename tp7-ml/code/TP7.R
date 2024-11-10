# Cargar librerías necesarias
library(dplyr)
library(ggplot2)
library(rpart)
library(readr)
library(knitr)
library(ROSE) # Para over/under sampling
library(tidyr)
library(randomForest)
library(xgboost)
library(caret)
library(e1071)
library(DMwR)

# 1. Dividir el dataset en train y validation
set.seed(123) # Para reproducibilidad
split_dataset <- function(data_path, train_ratio = 0.8) {
  # Leer el dataset
  df <- read.csv(data_path)
  
  # Crear índices aleatorios
  n <- nrow(df)
  train_size <- floor(train_ratio * n)
  indices <- sample(1:n, n)
  
  # Dividir el dataset
  train_data <- df[indices[1:train_size], ]
  validation_data <- df[indices[(train_size + 1):n], ]
  
  # Guardar los archivos
  write.csv(train_data, "../data/arbolado-mendoza-dataset-train.csv", row.names = FALSE)
  write.csv(validation_data, "../data/arbolado-mendoza-dataset-validation.csv", row.names = FALSE)
  
  return(list(train = train_data, validation = validation_data))
}

extract_data <- function(file_path, col_types){
  csv_data <- read_csv(file_path, col_types = col_types) 
  return(csv_data)
}

# 2. Análisis exploratorio
analyze_dataset <- function(train_data) {
  # a. Distribución de inclinación_peligrosa
  class_dist <- ggplot(train_data, aes(x = factor(inclinacion_peligrosa))) +
    geom_bar() +
    labs(title = "Distribución de inclinación peligrosa",
         x = "Inclinación peligrosa",
         y = "Cantidad de árboles")
  
  # b. Peligrosidad por sección
  section_danger <- train_data %>%
    group_by(nombre_seccion) %>%
    summarise(
      total_trees = n(),
      dangerous_trees = sum(inclinacion_peligrosa == 1),
      danger_ratio = dangerous_trees / total_trees
    ) %>%
    arrange(desc(danger_ratio))
  
  section_plot <- ggplot(section_danger, aes(x = reorder(nombre_seccion, -danger_ratio), y = danger_ratio)) +
    geom_bar(stat = "identity") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Ratio de peligrosidad por sección",
         x = "Sección",
         y = "Ratio de peligrosidad")
  
  # c. Peligrosidad por especie
  species_danger <- train_data %>%
    group_by(especie) %>%
    summarise(
      total_trees = n(),
      dangerous_trees = sum(inclinacion_peligrosa == 1),
      danger_ratio = dangerous_trees / total_trees
    ) %>%
    filter(total_trees >= 50) %>%  # Filtrar especies con pocos ejemplares
    arrange(desc(danger_ratio))
  
  species_plot <- ggplot(species_danger, aes(x = reorder(especie, -danger_ratio), y = danger_ratio)) +
    geom_bar(stat = "identity") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Ratio de peligrosidad por especie",
         x = "Especie",
         y = "Ratio de peligrosidad")
  
  return(list(
    class_distribution = class_dist,
    section_analysis = section_plot,
    species_analysis = species_plot,
    section_data = section_danger,
    species_data = species_danger
  ))
}

# 3. Histogramas y categorización
create_histograms <- function(train_data) {
  
  # b. Circunferencia del tronco
  circ_hist <- ggplot(train_data, aes(x = circ_tronco_cm)) +
    geom_histogram(bins = 30) +
    labs(title = "Distribución de circunferencia del tronco",
         x = "Circunferencia (cm)",
         y = "Frecuencia")
  
  # c. Circunferencia por clase
  circ_class_hist <- ggplot(train_data, aes(x = circ_tronco_cm, fill = factor(inclinacion_peligrosa))) +
    geom_histogram(position = "dodge", bins = 30) +
    labs(title = "Distribución de circunferencia por clase",
         x = "Circunferencia (cm)",
         y = "Frecuencia",
         fill = "Inclinación peligrosa")
  
  # d. Categorización de circunferencia
  quantiles <- quantile(train_data$circ_tronco_cm, probs = c(0.25, 0.5, 0.75))
  train_data$circ_tronco_cm_cat <- cut(train_data$circ_tronco_cm,
                                       breaks = c(-Inf, quantiles[1], quantiles[2], quantiles[3], Inf),
                                       labels = c("bajo", "medio", "alto", "muy alto"))
  
  write.csv(train_data, "../data/arbolado-mendoza-dataset-circ_tronco_cm-train.csv", row.names = FALSE)
  
  return(list(
    circ_histogram = circ_hist,
    circ_class_histogram = circ_class_hist,
    cutoff_points = quantiles
  ))
}

# 4. Clasificador aleatorio
random_classifier <- function(data) {
  # Generar probabilidades aleatorias
  data$prediction_prob <- runif(nrow(data))
  
  # Clasificar según el umbral
  data$prediction_class <- ifelse(data$prediction_prob > 0.5, 1, 0)
  
  return(data)
}

# 5. Clasificador por clase mayoritaria
biggerclass_classifier <- function(data) {
  # Determinar la clase mayoritaria
  majority_class <- as.numeric(names(which.max(table(data$inclinacion_peligrosa))))
  
  # Asignar la clase mayoritaria a todas las predicciones
  data$prediction_class <- majority_class
  
  return(data)
}

# 6. Métricas de evaluación
calculate_metrics <- function(true_labels, predicted_labels) {
  # Matriz de confusión
  TP <- sum(true_labels == 1 & predicted_labels == 1)
  TN <- sum(true_labels == 0 & predicted_labels == 0)
  FP <- sum(true_labels == 0 & predicted_labels == 1)
  FN <- sum(true_labels == 1 & predicted_labels == 0)
  
  # Cálculo de métricas
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  precision <- TP / (TP + FP)
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  
  # Crear matriz de confusión con nombres
  confusion_matrix <- matrix(c(TP, FN, FP, TN), nrow = 2, 
                             dimnames = list("Actual" = c("Positivo", "Negativo"),
                                             "Predicho" = c("Positivo", "Negativo")))
  
  # Retornar las métricas
  return(list(
    confusion_matrix = confusion_matrix,
    metrics = c(accuracy = accuracy,
                precision = precision,
                sensitivity = sensitivity,
                specificity = specificity)
  ))
}

# 7. Validación cruzada
create_folds <- function(data, k) {
  # Mezclar aleatoriamente los índices
  n <- nrow(data)
  indices <- sample(1:n, n)
  
  # Crear los folds
  fold_size <- floor(n/k)
  folds <- list()
  
  for(i in 1:k) {
    start_idx <- (i-1) * fold_size + 1
    end_idx <- min(i * fold_size, n)
    folds[[paste0("Fold", i)]] <- indices[start_idx:end_idx]
  }
  
  return(folds)
}

cross_validation <- function(data, k) {
  # Asegurarnos que inclinacion_peligrosa sea factor
  data$inclinacion_peligrosa <- as.factor(data$inclinacion_peligrosa)
  
  # Asegurarnos que seccion y especie sean factores
  data$seccion <- as.factor(data$seccion)
  data$especie <- as.factor(data$especie)
  
  # Crear los folds
  folds <- create_folds(data, k)
  
  # Almacenar métricas
  metrics <- matrix(0, nrow = k, ncol = 4)
  colnames(metrics) <- c("Accuracy", "Precision", "Sensitivity", "Specificity")
  
  # Fórmula para el árbol
  train_formula <- formula(inclinacion_peligrosa ~ altura + circ_tronco_cm + lat + long + seccion + especie)
  
  # Para cada fold
  for(i in 1:k) {
    # Separar datos de entrenamiento y prueba
    test_indices <- folds[[i]]
    train_indices <- unlist(folds[-i])
    
    train_data <- data[train_indices, ]
    test_data <- data[test_indices, ]
    
    # Entrenar modelo
    tree_model <- rpart(train_formula, 
                        data = train_data, 
                        method = "class",
                        control = rpart.control(minsplit = 20,
                                                minbucket = 7,
                                                cp = 0.01))
    
    # Predecir
    predictions <- predict(tree_model, test_data, type = 'class')
    
    # Asegurarnos que las predicciones y valores reales sean numéricos para calculate_metrics
    true_values <- as.numeric(as.character(test_data$inclinacion_peligrosa))
    pred_values <- as.numeric(as.character(predictions))
    
    # Calcular métricas
    fold_metrics <- calculate_metrics(true_values, pred_values)
    metrics[i,] <- fold_metrics$metrics
  }
  
  # Calcular media y desviación estándar
  results <- data.frame(
    Media = colMeans(metrics),
    DesvEstandar = apply(metrics, 2, sd)
  )
  
  return(results)
}

remove_rare_species <- function(data, min_count = 100) {
  # Contar frecuencia de cada especie
  species_counts <- data %>%
    count(especie) %>%
    filter(n >= min_count)
  
  # Filtrar dataset
  filtered_data <- data %>%
    filter(especie %in% species_counts$especie)
  
  return(filtered_data)
}

perform_oversampling <- function(data, method = "SMOTE") {
  # Convertir inclinacion_peligrosa a factor si no lo es
  if (!is.factor(data$inclinacion_peligrosa)) {
    data$inclinacion_peligrosa <- as.factor(data$inclinacion_peligrosa)
  }
  
  # Verificar que el método sea válido
  if (!(method %in% c("SMOTE", "ROSE"))) {
    stop("El método debe ser 'SMOTE' o 'ROSE'")
  }
  
  # Realizar oversampling usando el método seleccionado
  if (method == "SMOTE") {
    # Usando SMOTE de DMwR
    balanced_data <- DMwR::SMOTE(inclinacion_peligrosa ~ ., data = data)
  } else if (method == "ROSE") {
    # Usando ROSE para oversampling
    balanced_data <- ROSE::ovun.sample(
      inclinacion_peligrosa ~ .,
      data = data,
      method = "over",
      N = max(2 * nrow(data), nrow(data)) # Definir N si es necesario
    )$data
  }
  
  return(balanced_data)
}



perform_undersampling <- function(data) {
  # Convertir inclinacion_peligrosa a factor si no lo es
  data$inclinacion_peligrosa <- as.factor(data$inclinacion_peligrosa)
  
  # Realizar undersampling usando ROSE
  balanced_data <- ROSE::ovun.sample(
    inclinacion_peligrosa ~ .,
    data = data,
    method = "under",
    N = 2 * sum(data$inclinacion_peligrosa == levels(data$inclinacion_peligrosa)[1])
  )$data
  
  return(balanced_data)
}

perform_undersampling <- function(data) {
  # Convertir inclinacion_peligrosa a factor si no lo es
  data$inclinacion_peligrosa <- as.factor(data$inclinacion_peligrosa)
  
  # Identificar la clase minoritaria y su tamaño
  min_class_size <- min(table(data$inclinacion_peligrosa))
  
  # Definir N como el doble del tamaño de la clase minoritaria (para mayor balance)
  N <- 2 * min_class_size
  
  # Realizar undersampling usando ROSE
  balanced_data <- ROSE::ovun.sample(
    inclinacion_peligrosa ~ .,
    data = data,
    method = "under",
    N = N
  )$data
  
  return(balanced_data)
}

random_forest_classifier <- function(train_data) {
  # Convertir inclinacion_peligrosa a factor
  train_data$inclinacion_peligrosa <- as.factor(train_data$inclinacion_peligrosa)
  
  # Crear el modelo de Random Forest
  rf_model <- randomForest(inclinacion_peligrosa ~ altura + circ_tronco_cm + diametro_tronco + especie + seccion,
                           data = train_data,
                           importance = TRUE,
                           ntree = 500)  # Número de árboles
  
  return(rf_model)
}

predict_random_forest <- function(rf_model, validation_data) {
  predictions_prob <- predict(rf_model, validation_data, type = "prob")
  
  # Crear un data frame con id y probabilidad de la clase "peligroso" (1)
  result <- data.frame(
    id = validation_data$id,  # Asegúrate de que 'id' sea el nombre correcto de la columna
    inclinacion_peligrosa = predictions_prob[, "1"]  # Probabilidad de la clase "peligroso"
  )
  
  return(result)
}

ensemble_model <- function(datos, target_var, predictors) {
  
  # Convertir las variables categóricas a factores
  datos[[target_var]] <- as.factor(datos[[target_var]])
  
  # Dividir el dataset en conjunto de entrenamiento y prueba
  set.seed(123) # Para reproducibilidad
  index <- createDataPartition(datos[[target_var]], p = .8, list = FALSE)
  train_data <- datos[index, ]
  test_data <- datos[-index, ]
  
  # Entrenar el modelo de Random Forest
  rf_model <- randomForest(as.formula(paste(target_var, "~", paste(predictors, collapse = "+"))), 
                           data = train_data)
  
  # Entrenar el modelo SVM
  svm_model <- svm(as.formula(paste(target_var, "~", paste(predictors, collapse = "+"))), 
                   data = train_data)
  
  # Predicciones en el conjunto de prueba
  rf_pred <- predict(rf_model, newdata = test_data)
  svm_pred <- predict(svm_model, newdata = test_data)
  
  # Crear un dataframe con las predicciones
  combined_predictions <- data.frame(rf_pred, svm_pred)
  
  # Votación mayoritaria
  final_pred <- apply(combined_predictions, 1, function(x) {
    as.character(names(sort(table(x), decreasing = TRUE)[1]))
  })
  
  # Evaluar el modelo
  confusion_matrix <- confusionMatrix(factor(final_pred), test_data[[target_var]])
  
  return(confusion_matrix)
}

train_ensemble_model <- function(datos, target_var, predictors) {
  
  # Convertir las variables categóricas a factores
  datos[[target_var]] <- as.factor(datos[[target_var]])
  
  # Dividir el dataset en conjunto de entrenamiento y prueba
  set.seed(2001) # Para reproducibilidad
  index <- createDataPartition(datos[[target_var]], p = .8, list = FALSE)
  train_data <- datos[index, ]
  
  # Entrenar el modelo de Random Forest con parámetros ajustados
  rf_model <- randomForest(as.formula(paste(target_var, "~", paste(predictors, collapse = "+"))), 
                           data = train_data, 
                           ntree = 1000, 
                           mtry = 5) # num de variables a considerar
  
  # Entrenar el modelo SVM con parámetros ajustados
  svm_model <- svm(as.formula(paste(target_var, "~", paste(predictors, collapse = "+"))), 
                   data = train_data, 
                   cost = 20, 
                   gamma = 0.1, 
                   kernel = "sigmoid") # linear, polynomial, radial, sigmoid
  
  # Retornar los modelos entrenados
  return(list(rf_model = rf_model, svm_model = svm_model))
}

# Función para hacer predicciones con los modelos entrenados
predict_with_ensemble <- function(models, new_data) {
  # Asegurarse de que el nuevo dataset tenga la columna 'id'
  if (!"id" %in% colnames(new_data)) {
    stop("El nuevo dataset debe contener la columna 'id'.")
  }
  
  # Predicciones con el modelo de Random Forest
  rf_pred <- predict(models$rf_model, newdata = new_data)
  
  # Predicciones con el modelo SVM
  svm_pred <- predict(models$svm_model, newdata = new_data)
  
  # Crear un dataframe con las predicciones
  combined_predictions <- data.frame(rf_pred, svm_pred)
  
  # Votación mayoritaria
  final_pred_values <- apply(combined_predictions, 1, function(x) {
    as.character(names(sort(table(x), decreasing = TRUE)[1]))
  })
  
  # Crear el dataframe final con 'id' e 'inclinacion_peligrosa'
  final_pred <- data.frame(id = new_data$id, inclinacion_peligrosa = final_pred_values)
  
  return(final_pred)
}

