# LeNet 5 Test

### Introduzione

Piccola DNN con scopo puramente dimostrativo, sufficientemente flessibile da permettere alcuni esperimenti senza toccare neanche una riga di codice.

### Dataset preparation

Scarica e impacchetta il dataset da internet.

### Training

Il training viene wffettuato su un batch di 20 immagini per 19900 iterazioni. La struttura della rete è la seguente:

 1. **Convoluzione con ReLU**:
  * `x = [None, 28, 28, 1]`
  * `w = [5, 5, 1, 32]` inizializzato a `N(0, 0.1)`
  * Padding: `SAME`
  * Stride: `1`
  * `b = [32]` inizializzato a `0.1`
  * `y = [None, 28, 28, 32]`
  * `5 * 5 * 1 * 32 + 32 = 832` variabili
 2. **Maxpooling**:
  * `x = [None, 28, 28, 32]`
  * `k = [1, 2, 2, 1]`
  * Padding: `SAME`
  * Stride: `1`
  * `y = [None, 14, 14, 32]`
  * `0` variabili
 3. **Convoluzione con ReLU**:
  * `x = [None, 14, 14, 32]`
  * `w = [5, 5, 32, 64]` inizializzato a `N(0, 0.1)`
  * Padding: `SAME`
  * Stride: `1`
  * `b = [64]` inizializzato a `0.1`
  * `y = [None, 14, 14, 64]`
  * `5 * 5 * 32 * 64 + 64 = 51'264` variabili
 3. **Maxpooling**:
  * `x = [None, 14, 14, 64]`
  * `k = [1, 2, 2, 1]`
  * Padding: `SAME`
  * Stride: `1`
  * `y = [None, 7, 7, 64]`
  * `0` variabili
 5. **Fully connected layer**:
  * `x = [None, 7 * 7 * 64]`
  * `w = [7 * 7 * 64, 1024]`
  * `b = [1024]`
  * `y = [None, 1024]`
  * `7 * 7 * 64 * 1024 + 1024 = 3'212'288` variabili
 6. **Dropout**:
  * `x = y = [None, 1024]`
  * Probability is `0.5` in training, `1.0` in testing/validation
  * `0` variabili
 7. **Fully connected layer**:
  * `x = [None, 1024]`
  * `w = [1024, 10]`
  * `b = [10]`
  * `y = [None, 10]`
  * `1024 * 10 + 10 = 10'250` variabili

> In totale stiamo parlando di: `3'274'634`

### Test

Non c'è una vera e propria fase di validazione, si va direttamente al test su 3000 immagine (questo perchè non si è fatto hyperparameter tuning in questo caso molto speciale).

> RESULT ACCURACY: 95.4% on notMNIST small dataset, while large is used for training

### Todo

 * Migliorare la visualizzazione di quello che è stato addestrato dalla NN.
 * Esportare la versione trainata della rete.
