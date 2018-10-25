package com.alensalihbasic.wreckfacejavacv;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.text.InputType;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.EditText;
import android.widget.Toast;

public class TrainingInfo extends AppCompatActivity {

    private EditText mName;
    private SharedPreferences myPrefs;
    private SharedPreferences.Editor editor;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.info_training);

        mName = findViewById(R.id.train_name_et);
        mName.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS);
        mName.setImeOptions(EditorInfo.IME_ACTION_DONE);

        myPrefs = getApplicationContext().getSharedPreferences("myPrefs", MODE_PRIVATE);
        editor = myPrefs.edit();

        findViewById(R.id.train_next_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!mName.getText().toString().equals("")) {
                    Intent trainingActivityIntent = new Intent(TrainingInfo.this, TrainActivity.class);
                    String mNameString = mName.getText().toString().trim();
                    editor.putString("name", mNameString);
                    editor.apply();
                    startActivity(trainingActivityIntent);
                }else {
                    Toast.makeText(TrainingInfo.this, "Zaboravili ste upisati ime", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }
}
