def test_load():
  return 'loaded'

def prior_prob(full_table, the_column, the_column_value):
  assert the_column in full_table
  assert the_column_value in up_get_column(full_table, the_column)

  #your function body below
  t_list = up_get_column(full_table, the_column)
  p = sum([1 if v==the_column_value else 0 for v in t_list])/len(t_list)
  return p
  
def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]


def cond_probs_product(full_table, evidence_row, target_column, target_column_value):
  assert target_column in full_table
  assert target_column_value in up_get_column(full_table, target_column)
  assert isinstance(evidence_row, list)
  assert len(evidence_row) == len(up_list_column_names(full_table)) - 1   # - 1 because subtracting off the target column from full_table

  #your function body below
  table_columns = up_list_column_names(full_table)
  evidence_columns = table_columns[:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence_row)
  cond_prob_list = [cond_prob(full_table, e[0], e[1], target_column, target_column_value) for e in evidence_complete]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def cond_prob(full_table, the_evidence_column, the_evidence_column_value, the_target_column, the_target_column_value):
  assert the_evidence_column in full_table
  assert the_target_column in full_table
  assert the_evidence_column_value in up_get_column(full_table, the_evidence_column)
  assert the_target_column_value in up_get_column(full_table, the_target_column)

  #your function body below - copy and paste then align with parameter names
  t_subset = up_table_subset(full_table, the_target_column, 'equals', the_target_column_value)
  e_list = up_get_column(t_subset, the_evidence_column)
  p_b_a = sum([1 if v==the_evidence_column_value else 0 for v in e_list])/len(e_list)
  return p_b_a
  
def naive_bayes(full_table, evidence_row, target_column):
  assert target_column in full_table
  assert isinstance(evidence_row, list)
  assert len(evidence_row) == len(up_list_column_names(full_table)) - 1   # - 1 because subtracting off the target

  #compute P(target=0|...) by using cond_probs_product, finally multiply by P(target=0) using prior_prob
  target_val = 0
  neg1 = cond_probs_product(full_table, evidence_row, target_column, target_val)
  neg2=neg1*prior_prob(full_table, target_column, target_val)

  #do same for P(target=1|...)
  target_val = 1
  pos1 = cond_probs_product(full_table, evidence_row, target_column, target_val)
  pos2=pos1*prior_prob(full_table, target_column, target_val)

  #Use compute_probs to get 2 probabilities
  negative, positive = compute_probs(neg2,pos2)
  #return your 2 results in a list
  return [negative, positive]

def metrics(zipped_list):
  assert isinstance(zipped_list, list)
  assert all([isinstance(v, list) for v in zipped_list])
  assert all([len(v)==2 for v in zipped_list])
  assert all([isinstance(a,(int,float)) and isinstance(b,(int,float)) for a,b in zipped_list]), f'zipped_list contains a non-int or non-float'
  assert all([float(a) in [0.0,1.0] and float(b) in [0.0,1.0] for a,b in zipped_list]), f'zipped_list contains a non-binary value'

  #first compute the sum of all 4 cases. See code above
  sum_tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  sum_tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  sum_fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  sum_fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])

  #now can compute precicision, recall, f1, accuracy. Watch for divide by 0.
  precision = sum_tp/(sum_tp+sum_fp) if sum_tp+sum_fp else 0
  recall = sum_tp/(sum_tp+sum_fn) if sum_tp+sum_fn else 0
  f1_score_1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
  accuracy = (sum_tp+sum_tn)/(sum_tp+sum_tn+sum_fp+sum_fn)

  #now build dictionary with the 4 measures - round values to 2 places
  metrics_dict = {
  'Precision' : round(precision, 2),
  'Recall' : round(recall, 2),
  'F1' : round(f1_score_1, 2),
  'Accuracy' : round(accuracy, 2)}

  #finally, return the dictionary
  return metrics_dict

