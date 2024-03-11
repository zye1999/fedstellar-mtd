.PHONY: apidoc html clean all

apidoc:
	sphinx-apidoc -f -o . ../fedstellar

html:
	sphinx-build -b html . _build

clean:
	rm -rf _build

all: apidoc html
