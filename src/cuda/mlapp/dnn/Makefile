TARGETDIR := $(BINDIR)/$(BINSUBDIR)

all: dnntest

dnntest: dnntest.sh dnntest.py
	cp dnntest.sh dnntest.py $(TARGETDIR)
	chmod +x $(TARGETDIR)/dnntest.sh
	bash ./dnntest.sh

clean:
	rm $(TARGETDIR)/dnntest*