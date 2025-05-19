library("readxl")
library(readxl)
library(ggplot2)
library(dplyr)
library(car)
library(stats)
library(ggpubr)
library(ggridges)
library(Hmisc)
library(ggExtra)
library(cowplot)
library(dataPreparation)
library(tidyr)
library(scales)
library(plotly)
library(RColorBrewer)
library(pROC)
library(svglite) # para guardar plots como svg
library(Cairo)

# Cargamos la librería dplyr si aún no está cargada
if (!require(dplyr)) {
  install.packages("dplyr")
  library(dplyr)
}
knitr::opts_knit$set(dev.args = list(type = "cairo"))
windows.options(antialias = "cleartype")

df_decision <- read_excel("C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/resultados_machine_learning/IR_Analisis2_decision_scores.xlsx")
df_decision <- subset(df_decision, Modelo == "Logistic Regression")

df_shuffle <- read_excel("C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/resultados_machine_learning/IR_Analisis2_shuffle_decision_scores.xlsx")
df_shuffle <- subset(df_shuffle, Modelo == "Logistic Regression")

colnames(df_decision)[colnames(df_decision) == "Decision Score"] = "Decision_Score"
colnames(df_shuffle)[colnames(df_shuffle) == "Decision Score"] = "Decision_Score"

# Calculo la sensibilidad y especificidad para df_decision
myRoc_data_decision <- roc(response = df_decision$target, predictor = df_decision$Decision_Score, ci = TRUE, boot.n = 1000, ci.alpha = 0.95, stratified = FALSE)
ci_decision <- ci.se(myRoc_data_decision, specificities = seq(0, 1, 0.01))

# Convertir los nombres de las filas a una columna
ci_decision <- cbind(specificity = as.numeric(row.names(ci_decision)), ci_decision)
ci_decision <- data.frame(
  specificity = ci_decision[,1],
  sensitivity = ci_decision[,3],
  ci_low = ci_decision[,2],
  ci_high = ci_decision[,4]
)
# Asegurarse de que la columna 'specificity' sea numérica
ci_decision$specificity <- as.numeric(ci_decision$specificity)
# Realizar la operación matemática
ci_decision$specificity <- 1 - ci_decision$specificity
ci_decision$group <- sprintf("Autistic vs non-autistic children AUC = %.2f", myRoc_data_decision$auc)

# Calculo la sensibilidad y especificidad para df_shuffle
myRoc_data_shuffle <- roc(response = df_shuffle$target, predictor = df_shuffle$Decision_Score, ci = TRUE, boot.n = 1000, ci.alpha = 0.95, stratified = FALSE)
ci_shuffle <- ci.se(myRoc_data_shuffle, specificities = seq(0, 1, 0.01))
ci_shuffle <- cbind(specificity = as.numeric(row.names(ci_shuffle)), ci_shuffle)
ci_shuffle <- data.frame(
  specificity = ci_shuffle[,1],
  sensitivity = ci_shuffle[,3],
  ci_low = ci_shuffle[,2],
  ci_high = ci_shuffle[,4]
)
# Asegurarse de que la columna 'specificity' sea numérica
ci_shuffle$specificity <- as.numeric(ci_shuffle$specificity)
ci_shuffle$specificity <- 1 - ci_shuffle$specificity
ci_shuffle$group <- sprintf("Shuffle AUC = %.2f", myRoc_data_shuffle$auc)


# Combinamos los datos de ambas curvas
ROC_data <- rbind(ci_decision, ci_shuffle)

ROC_data$group <- factor(ROC_data$group, levels = c(sprintf("Autistic vs non-autistic children AUC = %.2f", myRoc_data_decision$auc), sprintf("Shuffle AUC = %.2f", myRoc_data_shuffle$auc)))

# Creamos el gráfico
ggplot(data = ROC_data, aes(x = specificity, y = sensitivity)) +
  geom_path(aes(col = group), size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high, fill = group), alpha = 0.2) +
  scale_color_manual(values = c("#C77CFF", "gray")) +
  scale_fill_manual(values = c("#C77CFF", "gray")) +
  theme(legend.position = c(0.60, 0.1),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        axis.title.x = element_text(size = 32, face = "bold"),
        axis.title.y = element_text(size = 32, face = "bold"),
        axis.text = element_text(size = 20, colour = "black", face = 'bold'),
        plot.title = element_text(size = 27, hjust = 0.48, face = 'bold'),
        legend.title = element_blank(),
        legend.text = element_text(size = 20, family = "Times", face = 'bold'),
        text = element_text(family = "Times")) +
  labs(x = "1-specificity", y = "Sensitivity") +
  geom_abline(intercept = 0, slope = 1, linetype = 2, size = 1)

ggsave(file = "C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/resultados_machine_learning/imagenes_roc_curve/roc_curve_IR_with_shuffle.svg", height = 6, width = 9)
