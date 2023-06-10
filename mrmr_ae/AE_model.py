from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
import tensorflow as tf


class Autoencoder:
    def __init__(
            self,
            input_dim,
            hidden_dim=30,
            activation='relu',
            regularizer=None,
            loss='mean_squared_error',
            learning_rate=0.0005,
            epochs=300,
            batch_size=32,
            initializer='glorot_uniform'
    ):
        """
        Autoencoder class for building and training an autoencoder model.

        Args:
            input_dim (int): Dimensionality of the input data.
            hidden_dim (int): Dimensionality of the hidden layer. Default is 30.
            activation (str): Activation function to use in the encoder and decoder layers. Default is 'relu'.
            regularizer (tf.keras.regularizers.Regularizer): Regularizer function to apply to the encoder layer. Default is None.
            loss (str): Loss function to use for training the autoencoder. Default is 'mean_squared_error'.
            learning_rate (float): Learning rate for the optimizer. Default is 0.0005.
            epochs (int): Number of epochs to train the autoencoder. Default is 300.
            batch_size (int): Batch size for training the autoencoder. Default is 32.
            initializer (str): Initializer for the weights of the encoder and decoder layers. Default is 'glorot_uniform'.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.regularizer = regularizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.initializer = initializer
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self._build_model()

    def _build_model(self):
        # Input layer
        input_layer = Input(shape=(self.input_dim,))

        # Encoder
        encoded = Dense(
            self.hidden_dim,
            activation=self.activation,
            kernel_initializer=self.initializer,
            activity_regularizer=self.regularizer
        )(input_layer)
        encoded = BatchNormalization()(encoded)

        # Decoder
        decoded = Dense(self.input_dim, activation=self.activation)(encoded)

        # Build encoder, decoder, and autoencoder models
        self.encoder = Model(inputs=input_layer, outputs=encoded)
        self.decoder = Model(inputs=self.encoder.input, outputs=decoded)
        self.autoencoder = Model(inputs=input_layer, outputs=decoded)

        # Compile the autoencoder model
        self.autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
            loss=self.loss
        )

    def fit(self, x_train, x_val=None):
        """
        Fit the autoencoder model to the training data.

        Args:
            x_train (ndarray): Training data.
            x_val (ndarray): Validation data. Optional. Default is None.
        """
        if x_val is not None:
            self.autoencoder.fit(
                x_train,
                x_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                validation_data=(x_val, x_val)
            )
        else:
            self.autoencoder.fit(
                x_train,
                x_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True
            )

    def extract_features(self, x):
        """
        Extract the features from the input data using the encoder model.

        Args:
            x (ndarray): Input data.

        Returns:
            ndarray: Extracted features.
        """
        return self.encoder.predict(x)

