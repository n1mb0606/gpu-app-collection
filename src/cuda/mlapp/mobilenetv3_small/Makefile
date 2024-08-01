TARGETDIR := $(BINDIR)/$(BINSUBDIR)

all: mobilenet

mobilenet: mobilenet.sh mobilenet.py
	cp mobilenet.sh mobilenet.py $(TARGETDIR)
	chmod +x $(TARGETDIR)/mobilenet.sh
	bash ./mobilenet.sh

clean:
	rm $(TARGETDIR)/mobilenet*