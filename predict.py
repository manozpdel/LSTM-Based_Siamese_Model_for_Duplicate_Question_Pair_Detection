from dependencies import *

def predict(question1, question2, threshold, model, verbose=False):
    generator = tf.data.Dataset.from_tensor_slices((([question1], [question2]),None)).batch(batch_size=1)
    
    
    # Call the predict method of your model and save the output into v1v2
    v1v2 = model.predict(generator)
    # Extract v1 and v2 from the model output
    _, n_feat = v1v2.shape
    v1 = v1v2[:, :int(n_feat/2)]
    v2 = v1v2[:, int(n_feat/2):]
    # Take the dot product to compute cos similarity of each pair of entries, v1, v2
    # Since v1 and v2 are both vectors, use the function tf.math.reduce_sum instead of tf.linalg.matmul
    d = tf.math.reduce_sum(v1 * v2)
    # Is d greater than the threshold?
    res = d > threshold

    
    if(verbose):
        print("Q1  = ", question1, "\nQ2  = ", question2)
        print("d   = ", d.numpy())
        print("res = ", res.numpy())

    return res.numpy()

# Make the predict function accessible for import
__all__ = ['predict']
