from sklearn.linear_model import LogisticRegression


def get_linearreg_model(class_weight='balanced'):
	model = LogisticRegression(class_weight=class_weight)
	return model
