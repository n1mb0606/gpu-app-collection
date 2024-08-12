TARGETDIR := $(BINDIR)/$(BINSUBDIR)

all: vit

vit: vit.sh vit.py
	cp vit.sh vit.py $(TARGETDIR)
	chmod +x $(TARGETDIR)/vit.sh
	bash ./vit.sh

clean:
	rm $(TARGETDIR)/vit*