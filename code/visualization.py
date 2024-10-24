import supervision as sv
def show_frank_results(result, size=(10,6)):
    detections = sv.Detections.from_ultralytics(result.result2)
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(
        scene=result.image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    sv.plot_image(annotated_image, size=size)
    if result.number:
        print(result.number)
    else:
        print('Не удалось считать показания!')