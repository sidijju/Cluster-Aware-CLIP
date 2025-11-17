SETUP_SCRIPT=scripts/setup_cpu.sh
COCO_SCRIPT=scripts/download_coco.sh

all: setup

setup:
	@echo "🔧 Running setup..."
	chmod +x $(SETUP_SCRIPT)
	./$(SETUP_SCRIPT)

coco:
	@echo "⬇️  Downloading MS COCO..."
	chmod +x $(COCO_SCRIPT)
	./$(COCO_SCRIPT)

# TODO: future datasets

clean:
	@echo "🧹 Cleaning temporary files..."
	rm -rf tmp/*

.PHONY: all setup coco imagenet clean
