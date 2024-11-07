freeze-extensions:
	code --list-extensions > extensions.txt

install-extensions:
	cat extensions.txt | xargs -L 1 code --install-extension