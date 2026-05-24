######################################################################
# PROGETTO CREDIT RISK - Reti Neurali
# Dataset: Sector_52.xlsx - Settore ATECO 52
# LB - NF
######################################################################

############################
# 0. SETUP INIZIALE
############################

remove(list = ls())

#install.packages(c("readxl", "fastDummies", "neuralnet", "pROC", "DescTools"))

library(readxl)
library(fastDummies)
library(neuralnet)
library(pROC)
library(DescTools)


############################
# 1. CARICAMENTO E ISPEZIONE
############################

Sector_52  <- read_excel("Sector_52.xlsx")
data_clean <- Sector_52

# Distribuzione della variabile target
table(data_clean$Default)
prop.table(table(data_clean$Default))
# 89.6% in bonis (0), 10.4% default (1)

# Valori mancanti
colSums(is.na(data_clean))

# Rimozione variabile costante: ateco2 assume sempre il valore "52"
unique(data_clean$ateco2)
data_clean$ateco2 <- NULL


############################
# GRAFICO 1 — Distribuzione della variabile Default (Class Imbalance)
############################

default_tab <- as.data.frame(table(data_clean$Default))
colnames(default_tab) <- c("Classe", "N")
default_tab$Perc  <- round(default_tab$N / sum(default_tab$N) * 100, 1)
default_tab$Label <- paste0(default_tab$N, "\n(", default_tab$Perc, "%)")
default_tab$Classe <- factor(default_tab$Classe,
                             labels = c("Non-Default (0)", "Default (1)"))

barplot(
  height = default_tab$N,
  names.arg = default_tab$Classe,
  col    = c("steelblue", "firebrick"),
  main   = "Distribuzione della variabile Default — Settore 52",
  ylab   = "Numero di osservazioni",
  ylim   = c(0, max(default_tab$N) * 1.15),
  border = NA,
  cex.names = 1.1
)
text(
  x   = c(0.7, 1.9),
  y   = default_tab$N + max(default_tab$N) * 0.04,
  labels = default_tab$Label,
  cex = 1.1,
  font = 2
)
mtext("Class imbalance: 89.6% In Bonis vs 10.4% Default",
      side = 3, line = 0.2, cex = 0.9, col = "gray40")


############################
# GRAFICO 2 — Boxplot variabili finanziarie chiave per classe di Default
############################

vars_box <- c("roa", "leverage", "current_ratio", "liquid_attivo", "autonomia_finanz", "mol_ricavi")
default_factor <- factor(data_clean$Default, labels = c("Non-Default", "Default"))

# Winsorizzazione al 5°-95° percentile per la sola visualizzazione
winsor_vis <- function(x) {
  q <- quantile(x, probs = c(0.05, 0.95), na.rm = TRUE)
  pmin(pmax(x, q[1]), q[2])
}

par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
for (v in vars_box) {
  y_vis <- winsor_vis(data_clean[[v]])
  boxplot(
    y_vis ~ default_factor,
    col    = c("steelblue", "firebrick"),
    main   = v,
    ylab   = "Valore (winsorizzato 5-95%)",
    xlab   = "",
    cex.main = 1.1,
    outline = FALSE
  )
}
par(mfrow = c(1, 1))


############################
# 2. CONVERSIONE VARIABILI CATEGORICHE
############################

# Le variabili categoriche vengono convertite in factor
data_clean$ateco       <- as.factor(data_clean$ateco)
data_clean$cod_regione <- as.factor(data_clean$cod_regione)
data_clean$socio       <- as.factor(data_clean$socio)
data_clean$Default     <- as.numeric(data_clean$Default)


############################
# 3. SPLIT STRATIFICATO TRAINING / TEST (2/3 - 1/3)
############################

# Lo split e' stratificato: i campioni vengono estratti separatamente
# per la classe 0 (in bonis) e la classe 1 (default).
# Con class imbalance (10% default) uno split casuale semplice potrebbe
# produrre un test set con pochissimi default, rendendo la valutazione
# delle performance inaffidabile.

set.seed(123)

indici_inbonis <- which(data_clean$Default == 0)
indici_default <- which(data_clean$Default == 1)

training_inbonis <- sample(indici_inbonis,
                           size    = round((2/3) * length(indici_inbonis)),
                           replace = FALSE)
training_default <- sample(indici_default,
                           size    = round((2/3) * length(indici_default)),
                           replace = FALSE)

indici_training <- c(training_inbonis, training_default)

training_raw <- data_clean[indici_training, ]
test_raw     <- data_clean[-indici_training, ]

# Verifica proporzioni
prop.table(table(training_raw$Default))
prop.table(table(test_raw$Default))

dim(training_raw)
dim(test_raw)


############################
# 4. SELEZIONE VARIABILI CON STEPWISE
############################

# La stepwise viene eseguita sul training set con le variabili nella loro
# forma originale (factor inclusi), prima di creare dummy e normalizzare.
# Si usano 4 valori di penalita' crescente.

modello_nullo    <- glm(Default ~ 1, data = training_raw,
                        family = binomial(link = "logit"))
modello_completo <- glm(Default ~ ., data = training_raw,
                        family = binomial(link = "logit"))

stepwise_k2  <- step(modello_nullo, scope = list(upper = modello_completo),
                     direction = "both", k = 2,  trace = FALSE)
stepwise_k3  <- step(modello_nullo, scope = list(upper = modello_completo),
                     direction = "both", k = 3,  trace = FALSE)
stepwise_k4  <- step(modello_nullo, scope = list(upper = modello_completo),
                     direction = "both", k = 4,  trace = FALSE)
stepwise_k10 <- step(modello_nullo, scope = list(upper = modello_completo),
                     direction = "both", k = 10, trace = FALSE)

summary(stepwise_k2)
summary(stepwise_k3)
summary(stepwise_k4)
summary(stepwise_k10)

# Variabili selezionate per ciascun modello
variabili_k2  <- attr(terms(stepwise_k2),  "term.labels")
variabili_k3  <- attr(terms(stepwise_k3),  "term.labels")
variabili_k4  <- attr(terms(stepwise_k4),  "term.labels")
variabili_k10 <- attr(terms(stepwise_k10), "term.labels")

print(variabili_k2)
print(variabili_k3)
print(variabili_k4)
print(variabili_k10)

# Frequenza di comparsa nei 4 modelli
tutte_variabili <- c(variabili_k2, variabili_k3, variabili_k4, variabili_k10)
frequenza_vars  <- sort(table(tutte_variabili), decreasing = TRUE)
print(frequenza_vars)

# Set finale: variabili presenti in almeno 2 stepwise su 4.
# La soglia >= 2 (anziche' >= 3) e' necessaria per ottenere circa 12 variabili
# in input, come richiesto. Con >= 3 il numero risultava troppo basso (6 variabili).
variabili_finali <- names(frequenza_vars[frequenza_vars >= 2])
cat("Numero di variabili selezionate:", length(variabili_finali), "\n")
print(variabili_finali)


############################
# 5. PREPROCESSING
############################

# Winsorization: taglia i valori estremi al 1° e 99° percentile.
# I percentili sono calcolati SOLO sul training e applicati al test
# per evitare data leakage.
winsor_fn <- function(vettore_train, vettore_test, probs = c(0.01, 0.99)) {
  q <- quantile(vettore_train, probs = probs, na.rm = TRUE)
  list(
    train = pmin(pmax(vettore_train, q[1]), q[2]),
    test  = pmin(pmax(vettore_test,  q[1]), q[2])
  )
}

# Normalizzazione min-max: porta ogni variabile nell'intervallo [0,1].
# Min e max calcolati sul training, applicati al test.
normalize_fn <- function(vettore_train, vettore_test) {
  minimo  <- min(vettore_train, na.rm = TRUE)
  massimo <- max(vettore_train, na.rm = TRUE)
  if (massimo == minimo) {
    list(train = rep(0, length(vettore_train)),
         test  = rep(0, length(vettore_test)))
  } else {
    list(
      train = (vettore_train - minimo) / (massimo - minimo),
      test  = (vettore_test  - minimo) / (massimo - minimo)
    )
  }
}

# Funzione di preprocessing completa:
# (a) subset sulle variabili selezionate
# (b) dummy per le categoriche
# (c) allineamento colonne training/test
# (d) conversione integer -> numeric
# (e) winsorization + normalizzazione sui soli predittori continui

prepara_dati_nn <- function(dati_training, dati_test, variabili_selezionate) {
  
  training <- dati_training[, c("Default", variabili_selezionate)]
  test     <- dati_test[,     c("Default", variabili_selezionate)]
  
  # Dummy per le variabili categoriche
  variabili_categoriche <- names(training)[sapply(training, is.factor)]
  if (length(variabili_categoriche) > 0) {
    training <- dummy_cols(training, select_columns = variabili_categoriche,
                           remove_selected_columns = TRUE,
                           remove_first_dummy = FALSE)
    test     <- dummy_cols(test,     select_columns = variabili_categoriche,
                           remove_selected_columns = TRUE,
                           remove_first_dummy = FALSE)
  }
  
  # Allineamento colonne: il test potrebbe non avere tutte le categorie del training
  for (v in setdiff(names(training), names(test)))  test[[v]]     <- 0
  for (v in setdiff(names(test), names(training)))  training[[v]] <- 0
  test <- test[, names(training)]
  
  # Conversione integer -> numeric (da Lesson 5)
  training[] <- lapply(training, function(x) if (is.integer(x)) as.numeric(x) else x)
  test[]     <- lapply(test,     function(x) if (is.integer(x)) as.numeric(x) else x)
  
  # Winsorization + normalizzazione sui soli predittori continui
  # Le dummy 0/1 vengono escluse perche' gia' nell'intervallo corretto
  predittori      <- setdiff(names(training), "Default")
  pred_continui   <- predittori[!sapply(training[predittori],
                                        function(x) all(na.omit(x) %in% c(0, 1)))]
  
  for (v in pred_continui) {
    w            <- winsor_fn(training[[v]], test[[v]])
    training[[v]] <- w$train
    test[[v]]     <- w$test
    n            <- normalize_fn(training[[v]], test[[v]])
    training[[v]] <- n$train
    test[[v]]     <- n$test
  }
  
  list(training = training, test = test)
}


############################
# 6. PREPARAZIONE DATASET FINALE
############################

dati_pronti  <- prepara_dati_nn(training_raw, test_raw, variabili_finali)
training_nn  <- dati_pronti$training
test_nn      <- dati_pronti$test

dim(training_nn)
dim(test_nn)

# make.names() evita problemi con nomi colonne non standard
names(training_nn) <- make.names(names(training_nn))
names(test_nn)     <- make.names(names(test_nn))

# Lista predittori (esclude il target)
predittori_nn <- setdiff(names(training_nn), "Default")


formula_nn <- as.formula(paste("Default ~", paste(predittori_nn, collapse = " + ")))

cat("Predittori in input:", length(predittori_nn), "\n")
print(predittori_nn)


############################
# 7. STIMA DELLE 4 RETI NEURALI
############################

set.seed(123)

neuralnet_1 <- neuralnet(
  formula_nn,
  data    = training_nn,
  hidden  = c(2),
  act.fct = "logistic",
  linear.output=FALSE,
  stepmax = 1e6
)

neuralnet_2 <- neuralnet(
  formula_nn,
  data    = training_nn,
  hidden  = c(3,1),
  act.fct = "logistic",
  linear.output=FALSE,
  stepmax = 1e6
)

neuralnet_3 <- neuralnet(
  formula_nn,
  data    = training_nn,
  hidden  = c(4,2),
  act.fct = "logistic",
  linear.output=FALSE,
  stepmax = 1e6
)

neuralnet_4 <- neuralnet(
  formula_nn,
  data    = training_nn,
  hidden  = c(17,8,5,4,1),
  act.fct = "logistic",
  linear.output=FALSE,
  stepmax = 1e6
)

# Grafico della struttura di ciascuna rete

plot(neuralnet_1)

plot(neuralnet_2, 
     radius          = 0.10,   # dimensione dei nodi (default 0.15)
     arrow.length    = 0.15,   # lunghezza delle frecce (default 0.2)
     intercept.factor = 0.15)  # dimensione dei nodi di intercetta

plot(neuralnet_3,
     radius          = 0.08,
     arrow.length    = 0.09,
     intercept.factor = 0.15)

plot(neuralnet_4,
     radius          = 0.07,
     arrow.length    = 0.09,
     intercept.factor = 0.15)

############################
# 8. RICERCA AUTOMATICA DEL CUT-OFF OTTIMALE SUL TEST SET
############################

# Variabili costanti per il calcolo dell'efficienza base
TOT_DEFAULT <- sum(test_nn$Default == 1)
TOT_INBONIS <- sum(test_nn$Default == 0)
inc_perf    <- (TOT_INBONIS * 0.05) - (0 * 0.45)
inc_worst   <- (0 * 0.05) - (TOT_DEFAULT * 0.45)

# Funzione che testa tutti i cut-off per trovare l'Expected Income massimo
ottimizza_cutoff <- function(modello, nome_modello) {
  
  probabilita <- as.vector(predict(modello, test_nn[, predittori_nn]))
  
  best_inc <- -Inf
  best_res <- NULL
  
  # Iteriamo su cut-off da 0.01 a 0.45 (step 0.01)
  for (c in seq(0.01, 0.45, by = 0.01)) {
    
    prev <- ifelse(probabilita > c, 1, 0)
    
    TP <- sum(test_nn$Default == 1 & prev == 1) # Default previsti correttamente (Rifiutati)
    FP <- sum(test_nn$Default == 0 & prev == 1) # In bonis previsti Default (Falsi allarmi, rifiutati)
    FN <- sum(test_nn$Default == 1 & prev == 0) # Default previsti In bonis (Falsi negativi, accettati -> PERDITA)
    TN <- sum(test_nn$Default == 0 & prev == 0) # In bonis previsti In bonis (Accettati -> GUADAGNO)
    
    acc <- mean(prev == test_nn$Default)
    inc <- (TN * 0.05) - (FN * 0.45)
    
    # Previene divisioni per zero se l'inc_worst dovesse essere uguale a inc_perf (improbabile ma sicuro)
    eff <- ifelse((inc_perf - inc_worst) == 0, 0, (inc - inc_worst) / (inc_perf - inc_worst))
    
    acc_rate <- (TN + FN) / nrow(test_nn)
    spread <- ifelse(TN == 0, 0, (FN * 0.45) / TN)
    
    # Se questo cut-off genera un Income maggiore, salva i risultati
    if (inc > best_inc) {
      best_inc <- inc
      best_res <- data.frame(
        model_name         = nome_modello,
        best_cutoff        = c,
        accuracy           = round(acc, 4),
        acceptance_rate    = round(acc_rate, 4),
        exp_income         = round(inc, 4),
        efficiency         = round(eff, 4),
        balancing_spread   = round(spread, 4),
        exp_income_perfect = round(inc_perf, 4),
        exp_income_worst   = round(inc_worst, 4)
      )
    }
  }
  return(best_res)
}

# Applichiamo la funzione alle 4 reti
results_rete1 <- ottimizza_cutoff(neuralnet_1, "Rete1_hidden")
results_rete2 <- ottimizza_cutoff(neuralnet_2, "Rete2_hidden")
results_rete3 <- ottimizza_cutoff(neuralnet_3, "Rete3_hidden")
results_rete4 <- ottimizza_cutoff(neuralnet_4, "Rete4_hidden")


############################
# 9. TABELLA FINALE DI CONFRONTO
############################

tabella_confronto <- rbind(results_rete1, results_rete2,
                           results_rete3, results_rete4)

# Ordinamento decrescente per Expected Income (dal migliore al peggiore)
tabella_confronto <- tabella_confronto[order(-tabella_confronto$exp_income), ]

print("--- RISULTATI CON CUT-OFF OTTIMIZZATO ---")
print(t(tabella_confronto))


############################
# 10. PLOT VISIVO DELLA RETE (Es. Rete 1)
############################

# 1. Ricalcoliamo le probabilità della Rete 1 per poterle disegnare
probabilita_rete1 <- as.vector(predict(neuralnet_1, test_nn[, predittori_nn]))

# 2. Peschiamo in automatico il cut-off perfetto che ha appena trovato la macchina
cutoff_ottimale <- results_rete1$best_cutoff

# 3. Creiamo il grafico
plot(probabilita_rete1, 
     main = 'Previsioni Rete Neurale 1 vs Cut-off Ottimizzato',
     ylab = 'Probabilità prevista di Default', 
     xlab = 'Indice Osservazione nel Test Set',
     pch = 16,
     col = "gray50")

# 4. Aggiungiamo la linea dinamica del cut-off ottimale
abline(h = cutoff_ottimale, lty = 2, lwd = 2, col = "red")

# 5. Aggiungiamo il testo dinamico che si aggiorna da solo
text(x = 10, y = cutoff_ottimale + 0.05, 
     labels = paste("Cut-off ottimizzato =", cutoff_ottimale), 
     col = "red", adj = 0)


############################
# GRAFICO 3 — Confronto Expected Income tra le 4 configurazioni
############################

nomi_reti <- c("Rete 1\nc(2)", "Rete 2\nc(3,1)", "Rete 3\nc(4,2)", "Rete 4\nc(17,8,5,4,1)")
income_reti <- c(results_rete1$exp_income, results_rete2$exp_income,
                 results_rete3$exp_income, results_rete4$exp_income)

# Colori: blu per reddito positivo, rosso per negativo
colori_barre <- ifelse(income_reti >= 0, "steelblue", "firebrick")

bp <- barplot(
  income_reti,
  names.arg = nomi_reti,
  col    = colori_barre,
  border = NA,
  main   = "Confronto Expected Income — 4 Configurazioni di Rete Neurale",
  ylab   = "Expected Income",
  ylim   = c(min(income_reti) * 1.25, max(income_reti) * 1.25),
  cex.names = 0.9
)
abline(h = 0, lwd = 1.5, col = "black")
text(
  x   = bp,
  y   = income_reti + sign(income_reti) * max(abs(income_reti)) * 0.04,
  labels = round(income_reti, 1),
  cex = 1.0,
  font = 2,
  col = colori_barre
)
mtext("Cut-off ottimizzato per ciascun modello | Tasso interesse 5% | LGD 45%",
      side = 3, line = 0.2, cex = 0.85, col = "gray40")


############################
# GRAFICO 4 — Confronto Efficiency tra le 4 configurazioni
############################

eff_reti <- c(results_rete1$efficiency, results_rete2$efficiency,
              results_rete3$efficiency, results_rete4$efficiency)

bp2 <- barplot(
  eff_reti * 100,
  names.arg = nomi_reti,
  col    = "steelblue",
  border = NA,
  main   = "Confronto Efficiency — 4 Configurazioni di Rete Neurale",
  ylab   = "Efficiency (%)",
  ylim   = c(0, max(eff_reti * 100) * 1.20),
  cex.names = 0.9
)
text(
  x   = bp2,
  y   = eff_reti * 100 + max(eff_reti * 100) * 0.03,
  labels = paste0(round(eff_reti * 100, 1), "%"),
  cex = 1.0,
  font = 2,
  col = "steelblue"
)
mtext("Efficiency = (EI_model - EI_worst) / (EI_perfect - EI_worst)",
      side = 3, line = 0.2, cex = 0.85, col = "gray40")

# Grafici per la differenziazione con le curva ROC
prob1 <- as.vector(predict(neuralnet_1, test_nn[, predittori_nn]))
prob2 <- as.vector(predict(neuralnet_2, test_nn[, predittori_nn]))
prob3 <- as.vector(predict(neuralnet_3, test_nn[, predittori_nn]))
prob4 <- as.vector(predict(neuralnet_4, test_nn[, predittori_nn]))

roc1 <- roc(test_nn$Default, prob1)
roc2 <- roc(test_nn$Default, prob2)
roc3 <- roc(test_nn$Default, prob3)
roc4 <- roc(test_nn$Default, prob4)

plot(roc1, col = "steelblue",  lwd = 2,
     main = "Curva ROC — Confronto 4 Reti Neurali",
     xlab = "False Positive Rate (1 - Specificità)",
     ylab = "True Positive Rate (Sensibilità)")
lines(roc2, col = "firebrick",   lwd = 2)
lines(roc3, col = "darkgreen",   lwd = 2)
lines(roc4, col = "darkorange",  lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray60")
legend("bottomright",
       legend = c(
         paste0("Rete 1 c(2)         AUC = ", round(auc(roc1), 3)),
         paste0("Rete 2 c(3,1)       AUC = ", round(auc(roc2), 3)),
         paste0("Rete 3 c(4,2)       AUC = ", round(auc(roc3), 3)),
         paste0("Rete 4 c(17,8,5,4,1) AUC = ", round(auc(roc4), 3))
       ),
       col = c("steelblue","firebrick","darkgreen","darkorange"),
       lwd = 2, cex = 0.85)

### Matrice di Confusione Rappresentazione Grafica ###

prob_rete1 <- as.vector(predict(neuralnet_1, test_nn[, predittori_nn]))
cutoff_rete1 <- results_rete1$best_cutoff
prev_rete1   <- ifelse(prob_rete1 > cutoff_rete1, 1, 0)

TP <- sum(test_nn$Default == 1 & prev_rete1 == 1)
FP <- sum(test_nn$Default == 0 & prev_rete1 == 1)
FN <- sum(test_nn$Default == 1 & prev_rete1 == 0)
TN <- sum(test_nn$Default == 0 & prev_rete1 == 0)

mat <- matrix(c(TN, FP, FN, TP), nrow = 2, byrow = TRUE)
etichette <- c(
  paste0("TN = ", TN, "\n(In bonis accettati\n✔ Guadagno)"),
  paste0("FP = ", FP, "\n(In bonis rifiutati\n– Lucro cessante)"),
  paste0("FN = ", FN, "\n(Default accettati\n✘ Perdita grave)"),
  paste0("TP = ", TP, "\n(Default rifiutati\n✔ Rischio evitato)")
)
colori_celle <- c("steelblue", "lightyellow", "firebrick", "lightgreen")

par(mar = c(4, 10, 4, 1))
plot(0, 0, type = "n", xlim = c(0, 2), ylim = c(0, 2),
     xaxt = "n", yaxt = "n", xlab = "", ylab = "",
     main = "Matrice di Confusione — Rete 1 (cut-off ottimizzato)")

for (i in 1:2) {
  for (j in 1:2) {
    idx <- (i - 1) * 2 + j
    rect(j - 1, 2 - i, j, 2 - i + 1,
         col = colori_celle[idx], border = "white", lwd = 3)
    text((j - 1) + 0.5, (2 - i) + 0.5,
         labels = etichette[idx], cex = 0.95, font = 2)
  }
}
axis(1, at = c(0.5, 1.5), labels = c("Previsto: In Bonis (0)", "Previsto: Default (1)"), tick = FALSE)
axis(2, at = c(0.5, 1.5), labels = c("Effettivo: Default (1)", "Effettivo: In Bonis (0)"),
     tick = FALSE, las = 2)

### Accuracy Confronto tra modelli  ###

accuracy_reti <- c(results_rete1$accuracy,
                   results_rete2$accuracy,
                   results_rete3$accuracy,
                   results_rete4$accuracy) * 100

income_reti_plot <- c(results_rete1$exp_income,
                      results_rete2$exp_income,
                      results_rete3$exp_income,
                      results_rete4$exp_income)

colori_punti <- c("steelblue","firebrick","darkgreen","darkorange")
nomi_plot    <- c("Rete 1\nc(2)", "Rete 2\nc(3,1)",
                  "Rete 3\nc(4,2)", "Rete 4\nc(17,8,5,4,1)")

plot(accuracy_reti, income_reti_plot,
     pch  = 19, cex  = 2.5,
     col  = colori_punti,
     xlim = c(60, 95),
     ylim = c(0, 4.5),
     main = "Trade-off: Accuracy vs Expected Income",
     xlab = "Accuracy (%)",
     ylab = "Expected Income")
abline(h = 0, lty = 2, col = "gray60")
text(accuracy_reti, income_reti_plot,
     labels = nomi_plot,
     pos = 3,offset = 1.5, cex = 0.85, col = colori_punti)

### Spread Reti Confronto ###

spread_reti <- c(results_rete1$balancing_spread,
                 results_rete2$balancing_spread,
                 results_rete3$balancing_spread,
                 results_rete4$balancing_spread) * 100

bp3 <- barplot(
  spread_reti,
  names.arg = nomi_reti,
  col    = c("steelblue","firebrick","darkgreen","darkorange"),
  border = NA,
  main   = "Balancing Spread — Tasso Minimo di Pareggio",
  ylab   = "Balancing Spread (%)",
  ylim   = c(0, max(spread_reti) * 1.25),
  cex.names = 0.9
)
abline(h = 5, lty = 2, lwd = 1.5, col = "black")
text(x = max(bp3) * 0.7, y = 5.15,
     labels = "Tasso di mercato = 5%", cex = 0.85, col = "black")
text(x   = bp3,
     y   = spread_reti + max(spread_reti) * 0.04,
     labels = paste0(round(spread_reti, 2), "%"),
     cex = 1.0, font = 2,
     col = c("steelblue","firebrick","darkgreen","darkorange"))
mtext("Più basso = maggiore vantaggio competitivo per la banca",
      side = 3, line = 0.2, cex = 0.85, col = "gray40")


### Plot Visivo della Rete con Cut-off al 0.02 con aziende in default e non colorate ###

# Colori diversi per default effettivo vs in bonis effettivo

colori_reali <- ifelse(test_nn$Default == 1, "firebrick", "steelblue")

plot(probabilita_rete1,
     main = "Previsioni Rete 1 — Probabilità di Default vs Cut-off Ottimizzato",
     ylab = "Probabilità prevista di Default",
     xlab = "Indice Osservazione nel Test Set",
     pch  = 16,
     col  = adjustcolor(colori_reali, alpha.f = 0.5),  # ← trasparenza
     cex  = 0.8)

abline(h = cutoff_ottimale, lty = 2, lwd = 2, col = "black")

text(x = 10, y = cutoff_ottimale + 0.04,
     labels = paste("Cut-off ottimizzato =", cutoff_ottimale),
     col = "black", adj = 0, cex = 0.9)

legend("topright",
       legend = c("In Bonis (effettivo)", "Default (effettivo)"),
       col    = c("steelblue", "firebrick"),
       pch    = 16, cex = 0.9,
       bg     = "white")  # ← legenda che spiega i colori

### Bar Plot Variabili Finanziari in Bonis e Default ###

par(mfrow = c(2, 3), mar = c(3, 4, 3, 1), oma = c(0, 0, 3, 0))
for (v in vars_box) {
  y_vis <- winsor_vis(data_clean[[v]])
  boxplot(
    y_vis ~ default_factor,
    col      = c("steelblue", "firebrick"),
    main     = v,
    ylab     = "",
    xlab     = "",
    cex.main = 1.2,
    outline  = FALSE,
    names    = c("In Bonis", "Default")  # ← etichette asse X leggibili
  )
}
mtext("Variabili Finanziarie Chiave: In Bonis vs Default (winsorizzato 5-95%)",
      outer = TRUE, cex = 1.1, font = 2)  # ← titolo generale sopra tutti
par(mfrow = c(1, 1))

colori_eff <- c("steelblue", "firebrick", "darkgreen", "darkorange")

### Confronto Efficienza Rete Nuerala Grafico Colorato  ###

bp2 <- barplot(
  eff_reti * 100,
  names.arg = nomi_reti,
  col       = colori_eff,       # ← colori diversi come negli altri grafici
  border    = NA,
  main      = "Confronto Efficiency — 4 Configurazioni di Rete Neurale",
  ylab      = "Efficiency (%)",
  ylim      = c(0, 100),        # ← scala fissa 0-100 più onesta
  cex.names = 0.9
)
abline(h = 100, lty = 2, lwd = 1.5, col = "gray60")
text(x = max(bp2) * 0.5, y = 97,
     labels = "Modello Perfetto = 100%",
     cex = 0.82, col = "gray40")
text(
  x      = bp2,
  y      = eff_reti * 100 + 2.5,
  labels = paste0(round(eff_reti * 100, 1), "%"),
  cex    = 1.0, font = 2,
  col    = colori_eff
)