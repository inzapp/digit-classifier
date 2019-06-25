import javafx.application.Application;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.shape.StrokeLineCap;
import javafx.stage.Stage;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;

import javax.imageio.ImageIO;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;
import java.util.logging.Level;
import java.util.logging.Logger;

class TableRowDataModel {
    private StringProperty n;
    private StringProperty result;

    TableRowDataModel(StringProperty n, StringProperty result) {
        this.n = n;
        this.result = result;
    }

    StringProperty nProperty() {
        return n;
    }

    StringProperty resultProperty() {
        return result;
    }
}

public class DigitClassifier extends Application implements Initializable {

    @FXML
    private Canvas canvas;

    @FXML
    private Button clearBt;

    @FXML
    private TableView table;

    @FXML
    private TableColumn<TableRowDataModel, String> nCol;

    @FXML
    private TableColumn<TableRowDataModel, String> resCol;

    private ANN_MLP model;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    @Override
    public void start(final Stage primaryStage) throws IOException {
        Parent fxml = FXMLLoader.load(getClass().getResource("gui.fxml"));
        primaryStage.setScene(new Scene(fxml));
        primaryStage.setTitle("Handwriting recognizer by inzapp");
        primaryStage.setResizable(false);
        primaryStage.setOnCloseRequest(e -> System.exit(0));
        primaryStage.show();
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        model = ANN_MLP.load("model.xml");
        nCol.setCellValueFactory(cell -> cell.getValue().nProperty());
        resCol.setCellValueFactory(cell -> cell.getValue().resultProperty());
        final GraphicsContext graphicsContext = canvas.getGraphicsContext2D();

        graphicsContext.setLineWidth(25);
        graphicsContext.setLineCap(StrokeLineCap.ROUND);
        canvas.addEventHandler(MouseEvent.MOUSE_PRESSED,
                event -> {
                    graphicsContext.beginPath();
                    graphicsContext.moveTo(event.getX(), event.getY());
                    graphicsContext.setStroke(Color.BLACK);
                    graphicsContext.stroke();
                });

        canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED,
                event -> {
                    graphicsContext.lineTo(event.getX(), event.getY());
                    graphicsContext.setStroke(Color.BLACK);
                    graphicsContext.stroke();
                });

        canvas.addEventHandler(MouseEvent.MOUSE_RELEASED,
                event -> {
                    File file = new File(new File("").getAbsolutePath() + "\\tmp.png");
                    try {
                        WritableImage writableImage = new WritableImage((int) canvas.getWidth(), (int) canvas.getHeight());
                        canvas.snapshot(null, writableImage);
                        RenderedImage renderedImage = SwingFXUtils.fromFXImage(writableImage, null);
                        ImageIO.write(renderedImage, "png", file);
                    } catch (IOException ex) {
                        Logger.getLogger(DigitClassifier.class.getName()).log(Level.SEVERE, null, ex);
                    }

                    Mat raw = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
                    Imgproc.resize(raw, raw, new Size(28, 28));
                    Core.bitwise_not(raw, raw);
//                    HighGui.imshow("img", raw);
//                    HighGui.waitKey(0);
//                    HighGui.destroyAllWindows();
                    raw.convertTo(raw, CvType.CV_32FC1);

                    raw = raw.reshape(1, 1);
                    Mat res = new Mat();
                    model.predict(raw, res);
                    Core.normalize(res, res, 0.0, 1.0, Core.NORM_MINMAX);
                    ObservableList<TableRowDataModel> observableList = FXCollections.observableArrayList();
                    double cur, max = -1;
                    int maxIdx = 0;
                    for (int i = 0; i < res.cols(); ++i) {
                        cur = res.get(0, i)[0] * 100;
                        observableList.add(new TableRowDataModel(new SimpleStringProperty(String.valueOf(i)),
                                new SimpleStringProperty(String.format("%.1f%%", cur))));
                        if (max < cur) {
                            max = cur;
                            maxIdx = i;
                        }
                    }

//                    table.requestFocus();
                    table.setItems(observableList);
                    table.getSelectionModel().select(maxIdx);
                    table.getSelectionModel().focus(maxIdx);
                });

        clearBt.setOnAction(event -> {
            ObservableList<TableRowDataModel> observableList = FXCollections.observableArrayList();
            for (int i = 0; i < 10; ++i) {
                observableList.add(new TableRowDataModel(new SimpleStringProperty(""),
                        new SimpleStringProperty("")));
            }
            table.setItems(observableList);
            canvas.getGraphicsContext2D().clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
        });
    }

    public static void main(String[] args) {
        launch(args);
    }
}