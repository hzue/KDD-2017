import numpy as np
import copy

class dataframe:

  @staticmethod
  def head(df, N):
    string = "\n\t"
    string += "\t".join(df.keys()) + "\n"
    for i in range(0, N):
      for k in df:
        string += "\t" + str(df[k][i])
      string += "\n"
    print(string)

  @staticmethod
  def tail(df, N):
    string = "\n\t"
    string += "\t".join(df.keys()) + "\n"
    for i in range(-N, 0):
      for k in df:
        string += "\t" + str(df[k][i])
      string += "\n"
    print(string)

  @staticmethod
  def new_col(df, names, fill=np.nan):
    template = [ fill for i in range (0, len(df['tollgate_id'])) ]
    for name in names:
      df[name] = copy.deepcopy(template)
    return df

  @staticmethod
  def groupby(df, cols):
    groups = []; groups_ds_map = {}; groups_ind_map = {}; iterable_groups = []

    for ind in range(0, len(df[cols[0]])):
      key_str = "-".join(str(df[col][ind]) for col in cols)
      tmp = tuple(df[col][ind] for col in cols)
      if not key_str in groups_ds_map:
        groups.append(tmp)
        groups_ds_map[key_str] = tmp
        groups_ind_map[key_str] = [ind]
      else: groups_ind_map[key_str].append(ind)

    for key in groups_ind_map:
      iterable_groups.append((key, groups_ind_map[key]))

    return groups, iterable_groups, groups_ind_map

  @staticmethod
  def header(df):
    return list(df.keys())

  @staticmethod
  def concat(df_list, axis='row'):
    base_df = copy.deepcopy(df_list[0])
    for df in df_list[1:]:
      for key in base_df.keys(): base_df[key] += df[key]
    return base_df

  @staticmethod
  def append_by_index(df1, df2, index):
    for key in df1.keys():
      df1[key].append(df2[key][index])

  @staticmethod
  def length(df):
    return len(df[list(df.keys())[0]])


