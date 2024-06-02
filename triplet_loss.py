from dependencies import *

def TripletLossFn(v1, v2, margin=0.25):
   
    # Compute similarity scores between v1 and v2
    scores = tf.linalg.matmul(v2, v1, transpose_b=True)
    
    # Get batch size as float
    batch_size = tf.cast(tf.shape(v1)[0], scores.dtype)
    
    # Extract positive scores (diagonal elements)
    positive = tf.linalg.diag_part(scores)
    
    # Compute negative scores, setting the diagonal (positive scores) to zero
    negative_zero_on_duplicate = scores - tf.linalg.diag(positive)
    
    # Compute the mean of negative scores
    mean_negative = tf.math.reduce_sum(negative_zero_on_duplicate, axis=1) / (batch_size - 1)
    
    # Mask to exclude positive scores
    mask_exclude_positives = tf.cast(
        (tf.eye(batch_size) == 1) | (negative_zero_on_duplicate > tf.expand_dims(positive, axis=1)),
        scores.dtype
    )
    
    # Adjust negatives to exclude positives
    negative_without_positive = negative_zero_on_duplicate - 2.0 * mask_exclude_positives
    
    # Get the closest negative score
    closest_negative = tf.math.reduce_max(negative_without_positive, axis=1)
    
    # Compute triplet loss components
    triplet_loss1 = tf.maximum(closest_negative - positive + margin, 0)
    triplet_loss2 = tf.maximum(mean_negative - positive + margin, 0)
    
    # Sum up the triplet loss components
    triplet_loss = tf.math.reduce_sum(triplet_loss1 + triplet_loss2)
    
    return triplet_loss

def TripletLoss(labels, out, margin=0.25):
   
    # Get embedding size
    _, out_size = out.shape
    
    # Split the embeddings into v1 and v2
    v1 = out[:, :int(out_size / 2)]
    v2 = out[:, int(out_size / 2):]
    
    return TripletLossFn(v1, v2, margin=margin)

# Make functions accessible for import
__all__ = ['TripletLossFn', 'TripletLoss']
