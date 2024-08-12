TARGETDIR := $(BINDIR)/$(BINSUBDIR)

all: vgg

vgg: vgg.sh vgg.py
	cp vgg.sh vgg.py $(TARGETDIR)
	chmod +x $(TARGETDIR)/vgg.sh
	bash ./vgg.sh

clean:
	rm $(TARGETDIR)/vgg*