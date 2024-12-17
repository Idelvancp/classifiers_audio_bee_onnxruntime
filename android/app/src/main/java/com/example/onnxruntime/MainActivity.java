package com.example.onnxruntime;

import io.flutter.embedding.android.FlutterActivity;
import io.flutter.embedding.engine.FlutterEngine;
import io.flutter.plugin.common.MethodChannel;
import androidx.annotation.NonNull;
import android.util.Log;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;

public class MainActivity extends FlutterActivity {
    private static final String CHANNEL = "com.example.audio/audio_processor";

    @Override
    public void configureFlutterEngine(@NonNull FlutterEngine flutterEngine) {
        super.configureFlutterEngine(flutterEngine);

        new MethodChannel(flutterEngine.getDartExecutor().getBinaryMessenger(), CHANNEL)
                .setMethodCallHandler(
                        (call, result) -> {
                            try {
                                switch (call.method) {
                                    case "onnxr":
                                        // Configurar o ambiente ONNX
                                        OrtEnvironment ortEnvironment = OrtEnvironment.getEnvironment();
                                        InputStream modelStream = getResources().openRawResource(R.raw.random_forest_classifier);
                                        byte[] modelBytes = new byte[modelStream.available()];
                                        modelStream.read(modelBytes);
                                        modelStream.close();
                                        OrtSession ortSession = ortEnvironment.createSession(modelBytes);

                                        // Ler os dados do arquivo CSV
                                        InputStream csvStream = getResources().openRawResource(R.raw.x_test_y_test);
                                        BufferedReader reader = new BufferedReader(new InputStreamReader(csvStream));
                                        List<float[]> inputDataList = new ArrayList<>();
                                        List<Integer> trueLabels = new ArrayList<>();

                                        String line;
                                        while ((line = reader.readLine()) != null) {
                                            String[] values = line.split(",");
                                            float[] sample = new float[values.length - 1];
                                            for (int i = 0; i < values.length - 1; i++) {
                                                sample[i] = Float.parseFloat(values[i]);
                                            }
                                            inputDataList.add(sample);
                                            trueLabels.add(Integer.parseInt(values[values.length - 1]));
                                        }
                                        reader.close();

                                        // Preparar os dados para o modelo
                                        int numSamples = inputDataList.size();
                                        int numFeatures = inputDataList.get(0).length;
                                        float[] flatInputData = new float[numSamples * numFeatures];

                                        for (int i = 0; i < numSamples; i++) {
                                            System.arraycopy(inputDataList.get(i), 0, flatInputData, i * numFeatures, numFeatures);
                                        }

                                        FloatBuffer inputBuffer = FloatBuffer.allocate(flatInputData.length);
                                        inputBuffer.put(flatInputData);
                                        inputBuffer.rewind();

                                        OnnxTensor tensor = OnnxTensor.createTensor(ortEnvironment, inputBuffer, new long[]{numSamples, numFeatures});

                                        // Fazer a predição
                                        String inputName = ortSession.getInputNames().iterator().next();
                                        Result output = ortSession.run(Map.of(inputName, tensor));

                                        // Extração dos valores previstos
                                        long[] predictions = (long[]) output.get(0).getValue();
                                        System.out.println("Predições: " + java.util.Arrays.toString(predictions));

                                        // Calcular métricas de avaliação
                                        int tp = 0, fp = 0, fn = 0, tn = 0;
                                        for (int i = 0; i < predictions.length; i++) {
                                            int actual = trueLabels.get(i);
                                            int predicted = (int) predictions[i];

                                            if (predicted == 1 && actual == 1) tp++;
                                            else if (predicted == 1 && actual == 0) fp++;
                                            else if (predicted == 0 && actual == 1) fn++;
                                            else if (predicted == 0 && actual == 0) tn++;
                                        }

                                        double precision = tp / (double) (tp + fp);
                                        double recall = tp / (double) (tp + fn);
                                        double f1Score = 2 * (precision * recall) / (precision + recall);
                                        double accuracy = (tp + tn) / (double) (tp + fp + fn + tn);

                                        System.out.println("Acurácia: " + accuracy);
                                        System.out.println("Precisão: " + precision);
                                        System.out.println("Recall: " + recall);
                                        System.out.println("F1-Score: " + f1Score);

                                        // Retornar o resultado
                                        result.success("Acurácia: " + accuracy + ", Precisão: " + precision + ", Recall: " + recall + ", F1-Score: " + f1Score);
                                        break;

                                    default:
                                        result.notImplemented();
                                        break;
                                }
                            } catch (Exception e) {
                                Log.e("MainActivity", "Erro ao executar tarefa", e);
                                result.error("TASK_ERROR", "Erro ao executar tarefa", e.getMessage());
                            }
                        }
                );
    }
}
