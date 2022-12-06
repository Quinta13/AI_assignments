from assignment_2.digits_classifiers.io.read_datasets import read_datasets


def main():
    X, y = read_datasets()
    print(len(X))


if __name__ == "__main__":
    main()
