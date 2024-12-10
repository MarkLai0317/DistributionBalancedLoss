from typing import Dict, List
import statistics
import argparse
import pandas as pd
import os

TRAIN_DATA_FILE_NAME = 'train_data.csv'

# group test dataset as well

class LabelGrouper:

    def __init__(self, data_path = None, image_id_column_name: str = "ID", df = None):
        if data_path is not None:
            self.df = pd.read_csv(data_path)
        else:
            self.df  = df
        self.id_column_name = image_id_column_name


    def generate_groups_csv(self, output_dir: str) -> List[str]:
        label_groups_list: List[List[str]]  = self.get_label_group_list()
        
        group_path_list: List[str] = []
        for i in range(len(label_groups_list)):
            output_path = os.path.join(output_dir, f"group{i+1}_{TRAIN_DATA_FILE_NAME}" )
            group_path_list.append(output_path)
            self.group_to_csv(label_groups_list[i], output_path)
        
        return group_path_list 
    
    def generate_test_groups_csv(self, group_path_list: str, test_data_path: str, data_name: str = 'test_data.csv') -> List[str]:
        test_df = pd.read_csv(test_data_path, index_col=0)
        
        for group_path in group_path_list:
            train_df = pd.read_csv(group_path, index_col=0,nrows=0)
            test_df_partition = test_df[train_df.columns]
            test_df_partition.to_csv(group_path.replace(TRAIN_DATA_FILE_NAME, data_name))
            
    
            

    def get_label_group_list(self) -> List[List[str]]:
        label_count = self.get_label_count()
        label_groups_list: List[List[str]]  = self.group_label(label_count)
        return label_groups_list


    def group_label(self, label_count: Dict[str, int]) -> List[List[str]]:
        sorted_dict = dict(sorted(label_count.items(), key=lambda item: item[1], reverse=True))
        print(sorted_dict)
        group_list: List[List[str]] = []
        current_group: List[str] = []
        max_count = -1
        irlbl_list: List[float] = []
        for label, count in sorted_dict.items():
            
            if max_count == -1:
                max_count = count
            
            irlbl_list.append(max_count/count)
            mean_IRLbl = statistics.mean(irlbl_list)
            if len(irlbl_list) > 1:
                cvir = self.calculate_CVIR(irlbl_list)
            else:
                cvir = 0
            
            if mean_IRLbl > 1.5 or cvir > 0.2:
                group_list.append(current_group)
                
                print(f"group {len(group_list)}:")
                print(current_group, "\n")
                
                current_group = [label]
                max_count = count
                irlbl_list = [1]
                print(f"current label: {label} -> {count}, MeanIRLbl: {mean_IRLbl} , CVIR: {cvir}")
                
            else:
                current_group.append(label)
                print(f"current label: {label} -> {count}, MeanIRLbl: {mean_IRLbl} , CVIR: {cvir}")
    
        
        if len(current_group) != 0:
            
            group_list.append(current_group)
            
            print(f"group {len(group_list)}:")
            print(current_group, "\n")
            
        return group_list

    def calculate_CVIR(self, irlbl_list):
        cvir = statistics.stdev(irlbl_list)/statistics.mean(irlbl_list)
        return cvir




    def get_label_count(self):
        label_count = self.df.drop(columns=[self.id_column_name]).sum().to_dict()
        return label_count


    def group_to_csv(self, label_list: List[str], output_path: str) -> None:
        group_df = self.df[[self.id_column_name] + label_list]
        group_df.to_csv(output_path, index=False)
        
        


class NewLabelGrouper(LabelGrouper):
    def group_label(self, label_count):
        sorted_dict = dict(sorted(label_count.items(), key=lambda item: item[1], reverse=True))
        print(sorted_dict)
        group_list: List[List[str]] = []
        current_group: List[str] = []
        max_count = -1
        irlbl_list: List[float] = []

        current_best: List[str] = []
        current_tail_index = 0
        current_best_tail_index = -1



        label_count_tuple = [(label, count) for label, count in sorted_dict.items()]
        used_labels_count = 0
        
        while used_labels_count < len(label_count_tuple) :
            # print(f"current_tail_index: {current_tail_index}")
            label, count = label_count_tuple[current_tail_index]
            
            if max_count == -1:
                max_count = count
            
            # calculate meanIRLbl and CVIR
            irlbl_list.append(max_count/count)
            mean_IRLbl = statistics.mean(irlbl_list)
            if len(irlbl_list) > 1:
                cvir = self.calculate_CVIR(irlbl_list)
            else:
                cvir = 0

            
            current_group.append(label)
            
            if mean_IRLbl <= 1.5 and cvir <= 0.2:
                
                current_best = current_group.copy()
                current_best_tail_index = current_tail_index


            if mean_IRLbl > 1.5 or current_tail_index == len(label_count_tuple)-1: 
                # print("break constraint")
                group_list.append(current_best)
                used_labels_count += len(current_best)

                current_group = []
                current_best = []
                irlbl_list = []
                max_count = -1

                current_tail_index = current_best_tail_index + 1

            else:
                current_tail_index += 1

            # print(f"current label: {label} -> {count}, MeanIRLbl: {mean_IRLbl} , CVIR: {cvir}")
        # if len(current_best) != 0:
            
        #     group_list.append(current_best)
            
            # print(f"group {len(group_list)}:")
            # print(current_group, "\n")
            
        return group_list
    


class DFSLabelGrouper(LabelGrouper):
    def __init__(self, current_best_group_num: int , data_path: str = None, image_id_column_name: str = "ID", ):
        super().__init__(data_path, image_id_column_name)
        self.current_best_group_num = current_best_group_num
        self.group_end_index_stack = []
        self.best_end_index_stack = []

    def group_label(self, label_count: Dict[str, int]) -> List[List[str]]:
        sorted_dict = dict(sorted(label_count.items(), key=lambda item: item[1], reverse=True))
        valid_end_index_dict = self.generate_valid_combination(sorted_dict)
        # print(valid_end_index_dict)

        self.start_index_cache = {}
        self.dfs(valid_end_index_dict, 0, len(sorted_dict)-1)
        print("==============dfs complete================")
        
        label_name_list = list(sorted_dict.keys())
        best_group_list = []

        start_index = 0
        for end_index in self.best_end_index_stack:
            best_group_list.append(label_name_list[start_index:end_index+1])
            start_index = end_index + 1

        return best_group_list
    def generate_valid_combination(self, sorted_dict: Dict[str, int]) -> Dict[int, List[int]]:
        
        

        
        label_count_tuple_list = [(label, count) for label, count in sorted_dict.items()]
        label_count_num = len(label_count_tuple_list)
        

        valid_end_index_dict = {}

        for i in range(label_count_num):
            valid_end_index_dict[i] = [i]
            
            label, count = label_count_tuple_list[i]
            # print(f"{label} -> {count}")
            max_count = count
            irlbl_list = [1]
            
            for j in range(i+1, label_count_num):
                _, count = label_count_tuple_list[j]
                irlbl_list.append(max_count/count)
                mean_IRLbl = statistics.mean(irlbl_list)
                if len(irlbl_list) > 1:
                    cvir = self.calculate_CVIR(irlbl_list)
                else:
                    cvir = 0
                #print(f"{label_count_tuple_list[j][0]} -> {label_count_tuple_list[j][1]}, MeanIRLbl: {mean_IRLbl} , CVIR: {cvir}")
                
                if mean_IRLbl <= 1.5 and cvir <= 0.2:
                    valid_end_index_dict[i].append(j)
                    
        
        return valid_end_index_dict

    def dfs(self, valid_end_index_dict: Dict[int, List[int]], start_index: int, end_index: int):
        # print(start_index)
        if start_index in self.start_index_cache:
            print("activate caches")
            old_end_index_len = len(self.group_end_index_stack)
            self.group_end_index_stack += self.start_index_cache.get(start_index)
            self.cache_start_index_result(self.group_end_index_stack, old_end_index_len)
            
            if len(self.group_end_index_stack) < self.current_best_group_num or self.best_end_index_stack == []:
                
                self.best_end_index_stack = self.group_end_index_stack.copy()
                self.current_best_group_num = len(self.best_end_index_stack)
                
        if len(self.group_end_index_stack) >= self.current_best_group_num-1 :
            return
        if start_index > end_index:
            return
        
        for i in range(len(valid_end_index_dict[start_index])-1, -1, -1):
            self.group_end_index_stack.append(valid_end_index_dict[start_index][i])
            
            if valid_end_index_dict[start_index][i] == end_index:


                self.cache_start_index_result(self.group_end_index_stack, len(self.group_end_index_stack))
                
                if len(self.group_end_index_stack) < self.current_best_group_num or self.best_end_index_stack == []:
                    
                    self.best_end_index_stack = self.group_end_index_stack.copy()
                    self.current_best_group_num = len(self.best_end_index_stack)
                                  # print(self.best_end_index_stack)
            self.dfs(valid_end_index_dict, valid_end_index_dict[start_index][i]+1, end_index)
            self.group_end_index_stack.pop()
    def cache_start_index_result(self, best_end_index_stack: List[int], old_end_index_len: int ):
        for i in range(old_end_index_len):
            start_index = best_end_index_stack[i]+1
            remaining_len = len(best_end_index_stack[i+1:])
            if start_index in self.start_index_cache:
                if len(self.start_index_cache.get(start_index)) > remaining_len:
                    self.start_index_cache[start_index] =  best_end_index_stack[i+1:]
            else:
                self.start_index_cache[start_index] = (best_end_index_stack[i+1:])


class PrunedDFSLabelGrouper(LabelGrouper):

    def __init__(self, data_path: str = None, image_id_column_name: str = "ID"):
        super().__init__(data_path, image_id_column_name)
        self.data_path = data_path
        self.pre_grouper = NewLabelGrouper(data_path, image_id_column_name)
    def group_label(self, label_count: Dict[str, int]) -> List[List[str]]:
        current_group = self.pre_grouper.group_label(label_count)
        
        possible_better_group = DFSLabelGrouper(len(current_group), data_path=self.data_path, image_id_column_name=self.id_column_name).group_label(label_count)
        print("geedy_group_num:", len(current_group))
        print("better_group_num:", len(possible_better_group))
        if possible_better_group != []:
            return possible_better_group
        
        return current_group
    


class DPLabelGrouper(LabelGrouper):
    
    
    def group_label(self, label_count: Dict[str, int]) -> List[List[str]]:
        sorted_dict = dict(sorted(label_count.items(), key=lambda item: item[1], reverse=True))
        valid_start_index_dict = self.generate_valid_combination(sorted_dict)
        # print(valid_end_index_dict)

        least_num_group_start_index = self.dp(valid_start_index_dict)
        print("==============dp complete================")
        
        label_name_list = list(sorted_dict.keys())
        best_group_list = []

        least_num_group_start_index.append(len(label_name_list))
        for i in range(len(least_num_group_start_index)-1):
            best_group_list.append(label_name_list[least_num_group_start_index[i]:least_num_group_start_index[i+1]])
       
        print(best_group_list)
        return best_group_list
    
    def generate_valid_combination(self, sorted_dict: Dict[str, int]) -> Dict[int, List[int]]:
        
        

        
        label_count_tuple_list = [(label, count) for label, count in sorted_dict.items()]
        label_count_num = len(label_count_tuple_list)
        

        valid_start_index_list = [[] for i in range(label_count_num)]

        for i in range(label_count_num):
            
            _, count = label_count_tuple_list[i]
            # print(f"{label} -> {count}")
            max_count = count
            irlbl_list = []
            
            for j in range(i, label_count_num):
                _, count = label_count_tuple_list[j]
                irlbl_list.append(max_count/count)
                mean_IRLbl = statistics.mean(irlbl_list)
                if len(irlbl_list) > 1:
                    cvir = self.calculate_CVIR(irlbl_list)
                else:
                    cvir = 0
                #print(f"{label_count_tuple_list[j][0]} -> {label_count_tuple_list[j][1]}, MeanIRLbl: {mean_IRLbl} , CVIR: {cvir}")
                
                # if mean_IRLbl <= 1.5 and cvir <= 0.2:
                #     valid_start_index_list[j].append(i)


                if mean_IRLbl <= 2.5 and cvir <= 0.4:
                    valid_start_index_list[j].append(i)
                    

        
        
        return valid_start_index_list
    
# each cache[i] is the best list of start index (a list of group) that is the best solution for the  labels[:i] (include i)
# each start index represent the start of a group
# each group is represented by a start index
# example: [0, 3, 5] means the group is labels[0:3], labels[3:5], labels[5:]
    def dp(self, valid_start_index_list):
        self.cache = { (-1): []}
        
        for i in range(len(valid_start_index_list)):
            min_len = float('inf')
            for valid_start_index in valid_start_index_list[i]:
                temp_len = len(self.cache[valid_start_index-1]) + 1
                if temp_len <= min_len:
                    min_len = temp_len
                    self.cache[i] = self.cache[valid_start_index-1] + [valid_start_index]

        return self.cache[len(valid_start_index_list)-1]
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)   
    parser.add_argument('--image_id_name', type=str, default="ID")
    parser.add_argument('--test_csv_data_path', type=str, required=True)
    parser.add_argument('--valid_csv_data_path', type=str, default=None)

    # parser.add_argument('--normal_class', type=int, default=1, help='Normal class index')
    # parser.add_argument('--training_labels_path', type=str)
    args = parser.parse_args()
	
    # Load the data
    

    # Group labels based on IRLBL
    label_grouper = DPLabelGrouper(args.csv_data_path, args.image_id_name)
    groups_path = label_grouper.generate_groups_csv(args.output_dir)
    label_grouper.generate_test_groups_csv(groups_path, args.test_csv_data_path)
    if args.valid_csv_data_path is not None:
        label_grouper.generate_test_groups_csv(groups_path, args.valid_csv_data_path, "valid_data.csv")

    print(f"Grouped labels saved to:")
    for group_path in groups_path:
        print(group_path)
    

    # Save the grouped labels to a file
    # with open(os.path.join(args.output_dir, 'grouped_labels.txt'), 'w') as f:
    #     for group in group_list:
    #         f.write(' '.join(group) + '\n')



