# for line in data:
#     line = line.replace(',', ' ').replace(':', ' ').split()
#     if not line:
#         continue
#
#     match line[0]:
#         case 'Monkey':
#             index = int(line[1])
#             self.monkeys.append(Monke(index))
#         case 'Starting':
#             for i in line:
#                 try:
#                     self.monkeys[index].inventory.append(int(i))
#                 except ValueError:
#                     continue
#         case 'Test':
#             self.monkeys[index].test = int(line[-1])
#         case 'If' if line[1] == 'true':
#             self.monkeys[index].target_true = int(line[-1])
#         case 'If' if line[1] == 'false':
#             self.monkeys[index].target_false = int(line[-1])
#         case 'Operation':
#             self.monkeys[index].op = line[-2]
#             self.monkeys[index].op_val = line[-1]
