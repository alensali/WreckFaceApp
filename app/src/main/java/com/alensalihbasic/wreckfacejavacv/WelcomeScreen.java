package com.alensalihbasic.wreckfacejavacv;

import android.content.Intent;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.view.View;

public class WelcomeScreen extends AppCompatActivity {


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.screen_welcome);

        findViewById(R.id.gender_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent genderIntent = new Intent(WelcomeScreen.this, GenderRecognizer.class);
                startActivity(genderIntent);
            }
        });

        findViewById(R.id.age_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent ageIntent = new Intent(WelcomeScreen.this, AgeRecognizer.class);
                startActivity(ageIntent);
            }
        });

        findViewById(R.id.face_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent faceIntent = new Intent(WelcomeScreen.this, TrainingInfo.class);
                startActivity(faceIntent);
            }
        });
    }
}
