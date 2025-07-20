### 1


#################################
############ PARTE 1 ############ 
#################################

library(readxl)
datos <- read_excel("chicos25.xlsx")

# Visualizamos las primeras filas del dataset
head(datos)

# Tipo de variables
str(datos)

# Pasamos a factor las variables mio y ma
datos$mio<-as.factor(datos$mio)
datos$ma<-as.factor(datos$ma)

# Pasamos el id a texto
datos$id <- as.character(datos$id)

# Chequeamos los cambios
str(datos)

# Estadistica basica
summary(datos)

# Buscamos missing values
colSums(is.na(datos))

# GGpairs
library(GGally)
library(ggplot2)
ggpairs(datos[, -which(names(datos) == "id")])

# GGpairs con box-plot en la diagnoal

library(GGally)
library(ggplot2)
library(rlang)

## Función personalizada para la diagonal
diag_custom <- function(data, mapping, ...) {
  var <- as_name(mapping$x)
  
  if (is.numeric(data[[var]])) {
    ggplot(data = data, mapping = mapping) +
      geom_boxplot(fill = "grey", color = "black", ...) +
      theme_minimal()
  } else {
    ggally_barDiag(data, mapping, ...)
  }
}

## Usar directamente la expresión en ggpairs
ggpairs(
  datos[, -which(names(datos) == "id")],
  diag = list(continuous = diag_custom, discrete = diag_custom)
)

# Outliers

## Función para encontrar outliers segun IQR (buscamos las que estan a mas de 3xIQR desde la caja)
find_outliers_iqr <- function(data, column_name) {
  q1 <- quantile(data[[column_name]], 0.25, na.rm = TRUE)
  q3 <- quantile(data[[column_name]], 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 3 * iqr
  upper_bound <- q3 + 3 * iqr
  
  outliers <- data[data[[column_name]] < lower_bound | data[[column_name]] > upper_bound, ]
  if (nrow(outliers) > 0) {
    cat("Outliers for variable:", column_name, "\n")
    print(outliers[, c("id", column_name)])
    cat("\n")
  } else {
    cat("No outliers found for variable:", column_name, "\n\n")
  }
}

## Identificar columnas numéricas
numeric_cols <- names(datos)[sapply(datos, is.numeric)]

## Iterar sobre las columnas numéricas y buscar outliers
for (col in numeric_cols) {
  find_outliers_iqr(datos, col)
}

# Eliminar outlier
datos <- datos[datos$id != "259", ]

#################################
############ PARTE 2 ############ 
#################################

# Particion 70-30
set.seed(666)
train <- sample(1:nrow(datos), nrow(datos) * 0.70, replace=FALSE)
datos_train <- datos[train, ]
datos_test <- datos[-train, ]

cat("Dimensiones del set de entrenamiento:", dim(datos_train), "\n")
cat("Dimensiones del set de prueba:", dim(datos_test), "\n")


#################################
############ PARTE 3 ############ 
#################################

### 1. Modelo completo ###

modelo_full <- glm(mio ~ . - id, family = binomial(), data = datos_train)
summary(modelo_full)

# Extraer los coeficientes del modelo
coeficientes_full <- coef(modelo_full)

# Ecuacion logit
equation_logit_full <- "ln(P(mio=Si) / (1 - P(mio=Si))) = "
equation_logit_full <- paste0(equation_logit_full, signif(coeficientes_full[1], 4))  # Intercept

for (i in 2:length(coeficientes_full)) {
  term <- names(coeficientes_full)[i]
  coef_value <- signif(coeficientes_full[i], 4)
  sign <- if (coef_value >= 0) " + " else " - "
  equation_logit_full <- paste0(equation_logit_full, sign, abs(coef_value), " * ", term)
}

cat("Ecuación en escala logit:\n")
cat(equation_logit_full, "\n\n")

# Ecuacion en forma exponencial
equation_exp_full <- "P(mio=Si) / (1 - P(mio=Si)) = exp("
equation_exp_full <- paste0(equation_exp_full, signif(coeficientes_full[1], 4))

for (i in 2:length(coeficientes_full)) {
  term <- names(coeficientes_full)[i]
  coef_value <- signif(coeficientes_full[i], 4)
  sign <- if (coef_value >= 0) " + " else " - "
  equation_exp_full <- paste0(equation_exp_full, sign, abs(coef_value), " * ", term)
}
equation_exp_full <- paste0(equation_exp_full, ")")

cat("Ecuación en forma exponencial:\n")
cat(equation_exp_full, "\n\n")

# Mostrar Odds Ratios
odds_ratios <- exp(coeficientes_full)

cat("Odds Ratios:\n")
print(signif(odds_ratios, 4))


### 2. Modelo step ###

modelo_step <- step(modelo_full, direction = "both",trace=T) 
summary(modelo_step)

# Extraer los coeficientes del modelo
coeficientes_step <- coef(modelo_step)

# Ecuación logit
equation_logit_step <- "ln(P(mio=Si) / (1 - P(mio=Si))) = "
equation_logit_step <- paste0(equation_logit_step, signif(coeficientes_step[1], 4))  # Intercept

for (i in 2:length(coeficientes_step)) {
  term <- names(coeficientes_step)[i]
  coef_value <- signif(coeficientes_step[i], 4)
  sign <- if (coef_value >= 0) " + " else " - "
  equation_logit_step <- paste0(equation_logit_step, sign, abs(coef_value), " * ", term)
}

cat("Ecuación en escala logit:\n")
cat(equation_logit_step, "\n\n")

# Ecuación en forma exponencial
equation_exp_step <- "P(mio=Si) / (1 - P(mio=Si)) = exp("
equation_exp_step <- paste0(equation_exp_step, signif(coeficientes_step[1], 4))

for (i in 2:length(coeficientes_step)) {
  term <- names(coeficientes_step)[i]
  coef_value <- signif(coeficientes_step[i], 4)
  sign <- if (coef_value >= 0) " + " else " - "
  equation_exp_step <- paste0(equation_exp_step, sign, abs(coef_value), " * ", term)
}
equation_exp_step <- paste0(equation_exp_step, ")")

cat("Ecuación en forma exponencial:\n")
cat(equation_exp_step, "\n\n")

# Mostrar Odds Ratios
odds_ratios <- exp(coeficientes_step)

cat("Odds Ratios:\n")
print(signif(odds_ratios, 4))


### 3. Modelo a eleccion ###


# Primero buscamos el modelo optimo (por AIC) entre todos los posibles pero resulto ser el mismo que el modelo step

## Convertir a data.frame normal, sin id
datos_train_bin <- datos_train[, names(datos_train) != "id"]
datos_train_bin$mio <- as.numeric(datos_train_bin$mio) - 1  # pasar a 0 y 1
datos_train_bin <- datos_train_bin[, c(setdiff(names(datos_train_bin), "mio"), "mio")]
datos_train_bin <- as.data.frame(datos_train_bin)

library(bestglm)

bestglm(Xy = datos_train_bin,
        family = binomial,
        IC = "AIC",
        method = "exhaustive")

# Cambiamos el enfoque y realizamos un modelo lo mas sencillo posible, empleando una sola regresora
modelo_simple <- glm(mio ~ spheq, family = binomial(), data = datos_train)
summary(modelo_simple)

# Extraer los coeficientes del modelo
coeficientes_simple <- coef(modelo_simple)

# Ecuación logit
equation_logit_simple <- "ln(P(mio=Si) / (1 - P(mio=Si))) = "
equation_logit_simple <- paste0(equation_logit_simple, signif(coeficientes_simple[1], 4))  # Intercept

for (i in 2:length(coeficientes_simple)) {
  term <- names(coeficientes_simple)[i]
  coef_value <- signif(coeficientes_simple[i], 4)
  sign <- if (coef_value >= 0) " + " else " - "
  equation_logit_simple <- paste0(equation_logit_simple, sign, abs(coef_value), " * ", term)
}

cat("Ecuación en escala logit:\n")
cat(equation_logit_simple, "\n\n")

# Ecuación en forma exponencial
equation_exp_simple <- "P(mio=Si) / (1 - P(mio=Si)) = exp("
equation_exp_simple <- paste0(equation_exp_simple, signif(coeficientes_simple[1], 4))

for (i in 2:length(coeficientes_simple)) {
  term <- names(coeficientes_simple)[i]
  coef_value <- signif(coeficientes_simple[i], 4)
  sign <- if (coef_value >= 0) " + " else " - "
  equation_exp_simple <- paste0(equation_exp_simple, sign, abs(coef_value), " * ", term)
}
equation_exp_simple <- paste0(equation_exp_simple, ")")

cat("Ecuación en forma exponencial:\n")
cat(equation_exp_simple, "\n\n")

# Mostrar Odds Ratios
odds_ratios <- exp(coeficientes_simple)

cat("Odds Ratios:\n")
print(signif(odds_ratios, 4))


# Comparacion de modelos

## Devianzas
anova(modelo_simple, modelo_step, modelo_full, test = "Chisq")

## AIC
AIC(modelo_simple, modelo_step, modelo_full)

#################################
############ PARTE 4 ############ 
#################################

# a. Significatividad de las variables
## Wald
summary(modelo_step)

## Significatividad de las variables utilizando devianzas
modelo_step_sin_spheq <- glm(mio ~ al + acd + sporthr + ma, family = binomial(), data = datos_train)
modelo_step_sin_al <- glm(mio ~ spheq + acd + sporthr + ma, family = binomial(), data = datos_train)
modelo_step_sin_acd <- glm(mio ~ spheq + al + sporthr + ma, family = binomial(), data = datos_train)
modelo_step_sin_sporthr <- glm(mio ~ spheq + al + acd + ma, family = binomial(), data = datos_train)
modelo_step_sin_ma <- glm(mio ~ spheq + al + acd, family = binomial(), data = datos_train)

anova(modelo_step_sin_spheq,modelo_step,test = "Chisq")
anova(modelo_step_sin_al,modelo_step,test = "Chisq")
anova(modelo_step_sin_acd,modelo_step,test = "Chisq")
anova(modelo_step_sin_sporthr,modelo_step,test = "Chisq")
anova(modelo_step_sin_ma,modelo_step,test = "Chisq")

# b. Bondad de ajuste: Test de Hosmer-Lemeshow

levels(datos_train$mio)
datos_train_mio_bin <- ifelse(datos_train$mio == "Si", 1, 0)

library(ResourceSelection)
hoslem.test(datos_train_mio_bin, fitted(modelo_step), g = 10)

# d. multicolinealidad
library(car)
vif(modelo_step)

# e. Análisis de puntos influyentes:
cooks=cooks.distance(modelo_step)
plot(cooks.distance(modelo_step))
abline(h = 0.009280742, col = "red", lty = 2) # Umbral 4/n como linea horizontal

library(broom)
model.metrics_full <- augment(modelo_step)

# Distancia de cook > 4/n
which(model.metrics_full$.cooksd>0.009280742)

# Guardar los influyentes
influential_indices <- which(model.metrics_full$.cooksd > 0.009280742)

# Generar un dataset con los influyentes
influential_points_df <- datos_train[influential_indices, ]

# Mostrar el dataset
print(influential_points_df)

summary(influential_points_df)
summary(datos_train)


#################################
############ PARTE 5 ############ 
#################################

library(e1071)

modelo_bayes <- naiveBayes(mio ~.-id, data = datos_train,laplace=0) #laplace=1 para que lo habilite
modelo_bayes

#################################
############ PARTE 6 ############ 
#################################

### Prediccion con el modelo_step (regresion logistica) ###

library(ROCR)

# Optimización del umbral de prediccion por maximizacion de F1-score
## Obtener las probabilidades predichas sobre el test
probs_test <- predict(modelo_step, newdata = datos_test, type = "response")

## Convertir la variable real en test a 0/1
real_test <- ifelse(datos_test$mio == "Si", 1, 0)

## Crear objeto 'prediction' para ROCR
pred <- prediction(probs_test, real_test)

## Calcular precisión y recall en distintos thresholds
prec <- performance(pred, "prec")
rec  <- performance(pred, "rec")

## Obtener F1 para cada threshold, manejando NA
precision_vals <- prec@y.values[[1]]
recall_vals    <- rec@y.values[[1]]
thresholds     <- prec@x.values[[1]]

## F1 con control de NA
f1_vals <- mapply(function(p, r) {
  if (is.na(p) || is.na(r) || (p + r == 0)) return(NA)
  return(2 * p * r / (p + r))
}, precision_vals, recall_vals)

## Umbral con F1 máximo
best_idx <- which.max(f1_vals)
best_threshold <- thresholds[best_idx]
best_f1 <- f1_vals[best_idx]

cat("Mejor threshold:", round(best_threshold, 3), "\n")
cat("F1-score máximo:", round(best_f1, 4), "\n")


# Prediccion con el mejor umbral

library(caret)

## Predecir probabilidades en datos de test
probs_rlog_step <- predict(modelo_step, newdata = datos_test, type = "response")

## rear variable real.factor con niveles 0 y 1
real.factor <- factor(ifelse(datos_test$mio == "Si", 1, 0), levels = c(0, 1))

## Definir punto de corte
cutoff <- best_threshold

## Crear vector de clases predichas (factor con niveles 0 y 1)
predi.factor <- factor(ifelse(probs_rlog_step >= cutoff, 1, 0), levels = c(0, 1))

## Crear matriz de confusión con caret
conf_matrix <- confusionMatrix(predi.factor, real.factor, positive = "1")
print(conf_matrix)

## Calcular F1 manualmente
matriz_conf <- table(Predicho = predi.factor, Observado = real.factor)

TP <- matriz_conf["1", "1"]
FP <- matriz_conf["1", "0"]
FN <- matriz_conf["0", "1"]
TN <- matriz_conf["0", "0"]

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * precision * recall / (precision + recall)

cat("Precision :", round(precision, 4), "\n")
cat("Recall :", round(recall, 4), "\n")

cat("F1-score :", round(f1_score, 4), "\n")

# Curva ROC y AUC

library(ROCR)

## Probabilidades predichas en test con modelo_step
probs_rlog_step <- predict(modelo_step, newdata = datos_test, type = "response")

## Crear objeto prediction para ROCR
prediccion <- prediction(probs_rlog_step, real.factor)

## Calcular curva ROC
roc_modelo <- performance(prediccion, measure = "tpr", x.measure = "fpr")

## Graficar curva ROC
plot(roc_modelo, main = "Curva ROC modelo_step")
abline(a = 0, b = 1, lty = 2, col = "gray")

## Calcular AUC
AUC <- performance(prediccion, measure = "auc")@y.values[[1]]
cat("AUC:", round(AUC, 4), "\n")

### Prediccion con red bayesiana (naive Bayes) ###

library(e1071)
library(caret) 
library(ROCR) 

## Asegurarse que la variable respuesta tiene niveles adecuados
datos_test$mio <- factor(datos_test$mio, levels = c("No", "Si"))

## Predicción de probabilidades
probs_bayes <- predict(modelo_bayes, newdata = datos_test, type = "raw")

## Predicción de clases
pred_bayes <- predict(modelo_bayes, newdata = datos_test, type = "class")
pred_bayes <- factor(pred_bayes, levels = c("No", "Si"))

## Matriz de confusión
confusion.1 <- table(Predicho = pred_bayes, Observado = datos_test$mio)
cm <- confusionMatrix(confusion.1, positive = "Si")
print(cm)

## Extraer precisión y recall
precision <- cm$byClass["Precision"]
recall    <- cm$byClass["Recall"]

## Calcular F1-score
if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) {
  F1 <- 2 * precision * recall / (precision + recall)
} else {
  F1 <- NA
}
cat("F1-score =", F1, "\n")

# Curva ROC y AUC
prediccion1 <- prediction(probs_bayes[, "Si"], datos_test$mio)
roc_modelo_bayes <- performance(prediccion1, measure = "tpr", x.measure = "fpr")

plot(roc_modelo_bayes, main = "Curva ROC - Naive Bayes")
abline(a = 0, b = 1, lty = 2)

AUC.modelo_bayes <- performance(prediccion1, "auc")
cat("AUC =", AUC.modelo_bayes@y.values[[1]], "\n")

### Comparacion de modelos ###

#################################
######### Model Comparison ######
#################################

library(ROCR)
library(caret)
library(e1071)

### --- Modelo Step (Logística) ---

# Probabilidades predichas
probs_rlog_step_test <- predict(modelo_step, newdata = datos_test, type = "response")
real_test_factor <- factor(ifelse(datos_test$mio == "Si", 1, 0), levels = c(0, 1))

# Clases predichas con umbral óptimo
pred_rlog_step_test <- factor(ifelse(probs_rlog_step_test >= best_threshold, 1, 0), levels = c(0, 1))

# Matriz de confusión
conf_matrix_step <- confusionMatrix(pred_rlog_step_test, real_test_factor, positive = "1")

# Métricas
metrics_step <- data.frame(
  Model = "Logistic Regression (modelo_step)",
  Accuracy = conf_matrix_step$overall["Accuracy"],
  Sensitivity = conf_matrix_step$byClass["Sensitivity"],
  Specificity = conf_matrix_step$byClass["Specificity"],
  Precision = conf_matrix_step$byClass["Pos Pred Value"],
  Recall = conf_matrix_step$byClass["Recall"]
)
metrics_step$F1_Score <- 2 * metrics_step$Precision * metrics_step$Recall /
  (metrics_step$Precision + metrics_step$Recall)


### --- Modelo Naive Bayes ---

# Probabilidades y clases predichas
probs_bayes_test <- predict(modelo_bayes, newdata = datos_test, type = "raw")[, "Si"]
pred_bayes_test <- predict(modelo_bayes, newdata = datos_test, type = "class")

# Asegurar niveles correctos
datos_test$mio <- factor(datos_test$mio, levels = c("No", "Si"))
pred_bayes_test <- factor(pred_bayes_test, levels = c("No", "Si"))

# Matriz de confusión
conf_matrix_bayes <- confusionMatrix(pred_bayes_test, datos_test$mio, positive = "Si")

# Métricas
metrics_bayes <- data.frame(
  Model = "Naive Bayes (modelo_bayes)",
  Accuracy = conf_matrix_bayes$overall["Accuracy"],
  Sensitivity = conf_matrix_bayes$byClass["Sensitivity"],
  Specificity = conf_matrix_bayes$byClass["Specificity"],
  Precision = conf_matrix_bayes$byClass["Pos Pred Value"],
  Recall = conf_matrix_bayes$byClass["Recall"]
)
metrics_bayes$F1_Score <- 2 * metrics_bayes$Precision * metrics_bayes$Recall /
  (metrics_bayes$Precision + metrics_bayes$Recall)


### --- Tabla Comparativa ---

comparison_table <- rbind(metrics_step, metrics_bayes)


### --- Curva ROC Comparativa ---

# Objetos ROCR
prediccion_step_roc <- prediction(probs_rlog_step_test, real_test_factor)
prediccion_bayes_roc <- prediction(probs_bayes_test, real_test_factor)

roc_modelo_step <- performance(prediccion_step_roc, measure = "tpr", x.measure = "fpr")
roc_modelo_bayes <- performance(prediccion_bayes_roc, measure = "tpr", x.measure = "fpr")

# Plot
plot(roc_modelo_step, col = "blue", main = "ROC: Logistic Regression vs Naive Bayes", lwd = 2)
plot(roc_modelo_bayes, col = "red", add = TRUE, lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright", legend = c("Logistic Regression", "Naive Bayes"),
       col = c("blue", "red"), lty = 1, lwd = 2, cex = 0.8)

# AUC
auc_rlog_step <- performance(prediccion_step_roc, "auc")@y.values[[1]]
auc_bayes <- performance(prediccion_bayes_roc, "auc")@y.values[[1]]

comparison_table$AUC <- c(auc_rlog_step, auc_bayes)

### --- Mostrar Tabla Final ---
cat("\n--- Model Comparison Table (including AUC) ---\n")
print(comparison_table, row.names = FALSE)

