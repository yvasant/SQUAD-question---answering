def prepSQuAD():
	import json
	from nltk.tokenize import word_tokenize
	count = 0
	filenames = ['dev', 'train']
	for filename in filenames:
		fpr = open("C://Users//user//Desktop//project//"+filename+"-v1.1.json", 'r',encoding='utf-8')
		body = fpr.read()
		js = json.loads(body)
		fpw = open("C://Users//user//Desktop//project//"+filename+".txt", 'w',encoding='utf-8')
		for c in js["data"]:
			for p in c["paragraphs"]:
				context = p["context"].split(' ')
				context_char = list(p["context"])
				context_pos = {}
				for qa in p["qas"]:

					question = word_tokenize(qa["question"])

					if filename == 'train':
						for a in qa['answers']:
							answer = a['text'].strip()
							answer_start = int(a['answer_start'])

						#add '.' here, just because NLTK is not good enough in some cases
						answer_words = word_tokenize(answer+'.')
						if answer_words[-1] == '.':
							answer_words = answer_words[:-1]
						else:
							answer_words = word_tokenize(answer)

						prev_context_words = word_tokenize( p["context"][0:answer_start ] )
						left_context_words = word_tokenize( p["context"][answer_start:] )
						answer_reproduce = []
						for i in range(len(answer_words)):
							if i < len(left_context_words):
								w = left_context_words[i]
								answer_reproduce.append(w)
						join_a = ' '.join(answer_words)
						join_ar = ' '.join(answer_reproduce)

						#if not ((join_ar in join_a) or (join_a in join_ar)):
						if join_a != join_ar:
							#print join_ar
							#print join_a
							#print 'answer:'+answer
							count += 1

						fpw.write(' '.join(prev_context_words+left_context_words)+'\t')
						fpw.write(' '.join(question)+'\t')
						#fpw.write(join_a+'\t')

						pos_list = []
						for i in range(len(answer_words)):
							if i < len(left_context_words):
								pos_list.append(str(len(prev_context_words)+i+1))
						if len(pos_list) == 0:
							print (join_ar)
							print (join_a)
							print ('answer:'+answer)
						assert(len(pos_list) > 0)
						fpw.write(' '.join(pos_list)+'\n')
					else:
						fpw.write(' '.join(word_tokenize( p["context"]) )+'\t')
						fpw.write(' '.join(question)+'\n')

		fpw.close()
	print ('SQuAD preprossing finished!')
prepSQuAD()