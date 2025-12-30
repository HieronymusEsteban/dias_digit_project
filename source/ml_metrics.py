def get_classification_subsets_metrics(labels_results, var_name, pred_var_name):
    positive_bools = labels_results[var_name] == 1
    negative_bools = labels_results[var_name] == 0
    positive_pred_bools = labels_results[pred_var_name] == 1
    negative_pred_bools = labels_results[pred_var_name] == 0
    
    positives = labels_results[positive_bools]
    negatives = labels_results[negative_bools]
    true_positives = labels_results[positive_bools & positive_pred_bools]
    true_negatives = labels_results[negative_bools & negative_pred_bools]
    
    false_negatives = labels_results[positive_bools & negative_pred_bools]
    false_positives = labels_results[negative_bools & positive_pred_bools]

    sensitivity = true_positives.shape[0] / positives.shape[0]
    print('sensitivity:')
    print(sensitivity)
    
    specificity = true_negatives.shape[0] / negatives.shape[0]
    print('specificity:')
    print(specificity)

    subsets_and_metrics = (positives, negatives, true_positives, true_negatives, 
                           false_negatives, false_positives, sensitivity, specificity)
    
    return subsets_and_metrics

def plot_conf_matrix(labels_results, label, prediction, cases):
    true_positives, false_positives, true_negatives, false_negatives, positives, negatives = cases
    # Calculate confusion matrix
    cm = confusion_matrix(labels_results[label], labels_results[prediction])
    
    number_true_positives = true_positives.shape[0]
    number_false_positives = false_positives.shape[0]
    number_true_negatives = true_negatives.shape[0]
    number_false_negatives = false_negatives.shape[0]
    
    sensitivity = number_true_positives / positives.shape[0]
    specificity = number_true_negatives / negatives.shape[0]
    if (number_true_positives + number_false_positives) > 0:
        precision = number_true_positives / (number_true_positives + number_false_positives)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        precision = None
        f1_score = None
    if positives.shape[0] > 0:
        miss_rate = number_false_negatives / positives.shape[0]
    else:
        miss_rate = None
    
    print("Confusion Matrix:")
    
    plt.figure(figsize=(8,6))
    confusion_matrix_data = [[number_true_negatives, number_false_positives], 
                              [number_false_negatives, number_true_positives]]
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    print(f'True Positives: {number_true_positives}')
    print(f'False Positives: {number_false_positives}')
    print(f'True Negatives: {number_true_negatives}')
    print(f'False Negatives: {number_false_negatives}')
    print(f'\nSensitivity (Recall): {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')
    if precision is not None:
        print(f'Precision: {precision:.4f}')
        print(f'Miss Rate (False Negative Rate): {miss_rate:.4f}')
        print(f'F1 Score: {f1_score:.4f}')

def save_conf_matrix_tag(labels_results, label, prediction, cases, filename_tag):
    true_positives, false_positives, true_negatives, false_negatives, positives, negatives = cases
    # Calculate confusion matrix
    cm = confusion_matrix(labels_results[label], labels_results[prediction])
    
    number_true_positives = true_positives.shape[0]
    number_false_positives = false_positives.shape[0]
    number_true_negatives = true_negatives.shape[0]
    number_false_negatives = false_negatives.shape[0]
    
    sensitivity = number_true_positives / positives.shape[0]
    specificity = number_true_negatives / negatives.shape[0]
    
    if (number_true_positives + number_false_positives) > 0:
        precision = number_true_positives / (number_true_positives + number_false_positives)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        precision = None
        f1_score = None
        
    if positives.shape[0] > 0:
        miss_rate = number_false_negatives / positives.shape[0]
    else:
        miss_rate = None
    
    
    plt.figure(figsize=(15,8))
    gs = plt.GridSpec(1, 2, width_ratios=[2, 1])
    
    plt.subplot(gs[0])
    confusion_matrix_data = [[number_true_negatives, number_false_positives], 
                             [number_false_negatives, number_true_positives]]
    heatmap = sns.heatmap(confusion_matrix_data, annot=True, fmt='d', 
               xticklabels=['Predicted Negative', 'Predicted Positive'], 
               yticklabels=['Actual Negative', 'Actual Positive'],
               cbar_kws={'label': 'Number of Instances'})
    plt.title('Confusion Matrix')
    
    plt.subplot(gs[1])
    plt.axis('off')
    if precision is not None:
        metrics_text = (f'Performance Metrics:\n\n'
                       f'True Positives: {number_true_positives}\n'
                       f'False Positives: {number_false_positives}\n'
                       f'True Negatives: {number_true_negatives}\n'
                       f'False Negatives: {number_false_negatives}\n\n'
                       f'Sensitivity: {sensitivity:.4f}\n'
                       f'Specificity: {specificity:.4f}\n'
                       f'Precision: {precision:.4f}\n'
                       f'Miss Rate: {miss_rate:.4f}\n'
                       f'F1 Score: {f1_score:.4f}')
    else:
        metrics_text = (f'Performance Metrics:\n\n'
                       f'True Positives: {number_true_positives}\n'
                       f'False Positives: {number_false_positives}\n'
                       f'True Negatives: {number_true_negatives}\n'
                       f'False Negatives: {number_false_negatives}\n\n'
                       f'Sensitivity: {sensitivity:.4f}\n'
                       f'Specificity: {specificity:.4f}\n')
        
    plt.text(0, 0.5, metrics_text, fontsize=10, 
            verticalalignment='center')
    
    plt.suptitle('Photography Detection: Confusion Matrix and Performance Metrics Based on is_photo Label as Ground Truth', fontsize=16)
    plt.tight_layout()
    filename = 'conf_matrix_metrics_' + filename_tag + '.pdf'
    output_path = data_path / filename
    plt.savefig(output_path)
    plt.close()